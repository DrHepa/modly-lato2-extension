from __future__ import annotations

import importlib.util
import io
import json
import errno
import hashlib
import os
from pathlib import Path
import platform
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, redirect_stderr
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("modly_lato2_setup", PROJECT / "setup.py")
assert SPEC is not None and SPEC.loader is not None
setup = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = setup
SPEC.loader.exec_module(setup)

from lato2_modly import runtime


FINGERPRINT = {
    "implementation": "cpython",
    "version": [3, 11],
    "cache_tag": "cpython-311",
    "abiflags": "",
    "soabi": "cpython-311-x86_64-linux-gnu",
    "platform": "linux-x86_64",
    "machine": "x86_64",
    "pointer_bits": 64,
}


def fake_plan(*, exact: bool = False):
    return types.SimpleNamespace(
        profile="exact-upstream" if exact else "portable",
        system="linux",
        arch="x64",
        gpu_sm=86,
        torch_lane="cu124",
        attention_backend="flash_attn" if exact else "sdpa",
        install_native_stack=exact,
        support_level="toolchain-dependent" if exact else "compatibility",
    )


def fake_context(root: Path) -> object:
    return setup.SetupContext(
        python_exe=Path(sys.executable),
        ext_dir=root,
        gpu_sm=86,
        cuda_version=124,
        accelerator="cuda",
        platform_name="linux",
        arch="x64",
        payload={
            "python_exe": sys.executable,
            "ext_dir": str(root),
            "gpu_sm": 86,
            "cuda_version": 124,
            "accelerator": "cuda",
            "platform": "linux",
            "arch": "x64",
        },
        host_fingerprint=FINGERPRINT,
    )


class ParseTests(unittest.TestCase):
    def test_current_json_contract(self) -> None:
        payload = {"python_exe": "python", "ext_dir": "/extension", "gpu_sm": 86}
        self.assertEqual(setup.parse_args(["setup.py", json.dumps(payload)]), payload)

    def test_legacy_contract(self) -> None:
        payload = setup.parse_args(["setup.py", "python", "/extension", "89", "124"])
        self.assertEqual(payload["gpu_sm"], 89)
        self.assertEqual(payload["cuda_version"], 124)
        self.assertEqual(payload["accelerator"], "cuda")

    def test_rejects_extra_legacy_arguments(self) -> None:
        with self.assertRaises(setup.SetupFailure) as raised:
            setup.parse_args(["setup.py", "python", "/extension", "89", "124", "extra"])
        self.assertEqual(raised.exception.code, "SETUP_ARGUMENTS_INVALID")

    def test_rejects_non_object_json(self) -> None:
        with self.assertRaises(setup.SetupFailure) as raised:
            setup.parse_args(["setup.py", "[]"])
        self.assertEqual(raised.exception.code, "SETUP_JSON_INVALID")

    def test_main_returns_actionable_error_on_stderr(self) -> None:
        stream = io.StringIO()
        with (
            patch.object(
                setup,
                "run_setup",
                side_effect=setup.SetupFailure("TEST_FAILURE", "run Repair now"),
            ),
            redirect_stderr(stream),
        ):
            result = setup.main(["setup.py", "{}"])
        self.assertEqual(result, 1)
        self.assertIn("ERROR [TEST_FAILURE] run Repair now", stream.getvalue())


class ContextTests(unittest.TestCase):
    def test_normalizes_host_contract_and_requires_cpython_311(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            payload = {
                "python_exe": sys.executable,
                "ext_dir": str(root),
                "gpu_sm": 86,
                "cuda_version": 124,
                "accelerator": "cuda",
                "platform": sys.platform,
                "arch": platform.machine(),
            }
            with patch.object(setup, "interpreter_fingerprint", return_value=FINGERPRINT):
                context = setup.validate_context(payload, root)
            self.assertEqual(context.platform_name, setup.current_platform_name())
            self.assertEqual(context.arch, setup._normalize_arch(platform.machine()))
            self.assertEqual(context.host_fingerprint, FINGERPRINT)

    def test_rejects_wrong_python_abi(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            payload = {
                "python_exe": sys.executable,
                "ext_dir": str(root),
                "gpu_sm": 86,
                "platform": sys.platform,
                "arch": platform.machine(),
            }
            wrong = {**FINGERPRINT, "version": [3, 12]}
            with patch.object(setup, "interpreter_fingerprint", return_value=wrong):
                with self.assertRaises(setup.SetupFailure) as raised:
                    setup.validate_context(payload, root)
            self.assertEqual(raised.exception.code, "PYTHON_ABI_UNSUPPORTED")

    def test_interpreter_probe_does_not_inherit_secrets_or_python_overrides(self) -> None:
        completed = types.SimpleNamespace(stdout=json.dumps(FINGERPRINT))
        inherited = {
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": "/tmp/injected",
            "PYTHONHOME": "/tmp/python-home",
            "PIP_INDEX_URL": "https://evil.invalid/simple",
            "HF_TOKEN": "secret",
        }
        with patch.dict(setup.os.environ, inherited, clear=True), patch.object(
            setup.subprocess, "run", return_value=completed
        ) as run:
            self.assertEqual(
                setup.interpreter_fingerprint(Path(sys.executable)), FINGERPRINT
            )
        child_env = run.call_args.kwargs["env"]
        self.assertEqual(child_env, {"PATH": "/usr/bin:/bin"})


class DiskEstimateTests(unittest.TestCase):
    def test_accounts_for_complete_and_resumable_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            complete_data = b"a" * 20
            complete = types.SimpleNamespace(
                relative_path="ckpt/a.pt",
                size=20,
                sha256=hashlib.sha256(complete_data).hexdigest(),
            )
            partial = types.SimpleNamespace(
                relative_path="ckpt/b.pt",
                size=30,
                sha256=hashlib.sha256(b"b" * 30).hexdigest(),
            )
            missing = types.SimpleNamespace(
                relative_path="ckpt/c.pt",
                size=40,
                sha256=hashlib.sha256(b"c" * 40).hexdigest(),
            )
            (root / "ckpt").mkdir()
            (root / "ckpt/a.pt").write_bytes(complete_data)
            with (root / "ckpt/b.pt.part").open("wb") as handle:
                handle.truncate(7)
            with patch.object(setup, "ASSETS", (complete, partial, missing)):
                self.assertEqual(setup._remaining_asset_bytes(root), 23 + 40)

    def test_authenticated_asset_needs_no_growth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = b"a" * 20
            spec = types.SimpleNamespace(
                relative_path="ckpt/a.pt",
                size=20,
                sha256=hashlib.sha256(data).hexdigest(),
            )
            (root / "ckpt").mkdir()
            (root / "ckpt/a.pt").write_bytes(data)
            with patch.object(setup, "ASSETS", (spec,)):
                self.assertEqual(setup._remaining_asset_bytes(root), 0)

    def test_forged_ready_marker_cannot_hide_same_size_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = types.SimpleNamespace(
                relative_path="ckpt/a.pt",
                size=20,
                sha256=hashlib.sha256(b"expected".ljust(20, b"!")).hexdigest(),
            )
            (root / "ckpt").mkdir()
            (root / "ckpt/a.pt").write_bytes(b"corrupt".ljust(20, b"?"))
            (root / setup.READY_MARKER_FILENAME).write_text(
                json.dumps(
                    {
                        "schema_version": setup.READY_SCHEMA_VERSION,
                        "extension_id": setup.EXTENSION_ID,
                        "revision_id": setup.REVISION_ID,
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(setup, "ASSETS", (spec,)):
                self.assertEqual(setup._remaining_asset_bytes(root), 20)


class SetupLockTests(unittest.TestCase):
    @unittest.skipIf(sys.platform == "win32", "POSIX replacement regression")
    def test_writer_revalidates_lock_identity_after_acquisition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / setup.SETUP_LOCK_FILENAME
            lock.write_bytes(b"\0")
            replacement = root / "replacement.lock"
            replacement.write_bytes(b"\0")
            original_try = setup._try_setup_lock

            def acquire_then_replace(handle, platform_name):
                acquired = original_try(handle, platform_name)
                if acquired:
                    os.replace(replacement, lock)
                return acquired

            with (
                patch.object(setup, "_try_setup_lock", side_effect=acquire_then_replace),
                self.assertRaises(setup.SetupFailure) as raised,
            ):
                with setup.setup_lock(root, timeout=0):
                    self.fail("a replaced writer lock must never guard setup")
            self.assertEqual(raised.exception.code, "SETUP_LOCK_UNSAFE")

    def test_runtime_readers_share_the_lock_and_block_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with setup.setup_lock(root, timeout=0.2, poll_interval=0.01):
                pass
            with runtime._setup_read_lock(root, timeout=0.2, poll_interval=0.01):
                with runtime._setup_read_lock(root, timeout=0.2, poll_interval=0.01):
                    with self.assertRaises(setup.SetupFailure) as raised:
                        with setup.setup_lock(root, timeout=0, poll_interval=0.01):
                            self.fail("Repair must not mutate an active inference")
            self.assertEqual(raised.exception.code, "SETUP_BUSY")

    def test_hardlinked_setup_lock_is_rejected_without_modifying_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside.txt"
            lock = root / setup.SETUP_LOCK_FILENAME
            outside.write_bytes(b"")
            try:
                os.link(outside, lock)
            except OSError as exc:
                self.skipTest(f"hardlinks unavailable: {exc}")
            with self.assertRaises(setup.SetupFailure) as raised:
                with setup.setup_lock(root, timeout=0):
                    self.fail("a hardlinked lock must never be acquired")
            self.assertEqual(raised.exception.code, "SETUP_LOCK_UNSAFE")
            self.assertEqual(outside.read_bytes(), b"")

    def test_second_setup_times_out_with_actionable_busy_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with setup.setup_lock(root, timeout=0.2, poll_interval=0.01):
                with self.assertRaises(setup.SetupFailure) as raised:
                    with setup.setup_lock(root, timeout=0, poll_interval=0.01):
                        self.fail("a concurrent setup must not acquire the lock")
            self.assertEqual(raised.exception.code, "SETUP_BUSY")
            self.assertIn("Install or Repair", raised.exception.public_message)
            with setup.setup_lock(root, timeout=0.2, poll_interval=0.01):
                pass

    def test_windows_backend_uses_msvcrt_nonblocking_and_unlock_modes(self) -> None:
        calls: list[tuple[int, int, int]] = []
        fake_msvcrt = types.SimpleNamespace(
            LK_NBLCK=101,
            LK_UNLCK=102,
            locking=lambda fd, mode, count: calls.append((fd, mode, count)),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock"
            path.write_bytes(b"\0")
            with path.open("r+b") as handle, patch.dict(
                sys.modules, {"msvcrt": fake_msvcrt}
            ):
                self.assertTrue(setup._try_setup_lock(handle, "win32"))
                setup._release_setup_lock(handle, "win32")
                self.assertEqual(
                    [(mode, count) for _fd, mode, count in calls],
                    [(101, 1), (102, 1)],
                )

    def test_would_block_errno_is_retried_not_reported_as_io_failure(self) -> None:
        def busy(_fd, _mode, _count):
            raise OSError(errno.EACCES, "busy")

        fake_msvcrt = types.SimpleNamespace(
            LK_NBLCK=101,
            LK_UNLCK=102,
            locking=busy,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock"
            path.write_bytes(b"\0")
            with path.open("r+b") as handle, patch.dict(
                sys.modules, {"msvcrt": fake_msvcrt}
            ):
                self.assertFalse(setup._try_setup_lock(handle, "win32"))


class EnvironmentTests(unittest.TestCase):
    def test_install_space_sums_requirements_on_one_volume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            cache = root / "cache"
            cache.mkdir()
            with (
                patch.object(setup, "_volume_key", return_value=(1, "/")),
                patch.object(setup, "_require_free_space") as require,
            ):
                setup._preflight_install_storage(fake_context(root), fake_plan(), cache)
            require.assert_called_once()
            path, required, purpose = require.call_args.args
            self.assertEqual(path, root)
            self.assertEqual(
                required,
                setup.PORTABLE_ENVIRONMENT_FREE_BYTES
                + setup.PORTABLE_BUILD_CACHE_FREE_BYTES,
            )
            self.assertIn("Python environment", purpose)
            self.assertIn("build cache", purpose)

    def test_install_space_checks_split_volumes_independently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            cache = root / "cache"
            cache.mkdir()

            def volume(path: Path) -> tuple[int, str]:
                return (1, "/extensions") if path == root else (2, "/models")

            with (
                patch.object(setup, "_volume_key", side_effect=volume),
                patch.object(setup, "_require_free_space") as require,
            ):
                setup._preflight_install_storage(fake_context(root), fake_plan(), cache)
            self.assertEqual(require.call_count, 2)
            requirements = {call.args[0]: call.args[1] for call in require.call_args_list}
            self.assertEqual(requirements[root], setup.PORTABLE_ENVIRONMENT_FREE_BYTES)
            self.assertEqual(requirements[cache], setup.PORTABLE_BUILD_CACHE_FREE_BYTES)

    def test_reuse_rejects_venv_root_alias_and_invalidates_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            external = root / "external-venv"
            (external / "bin").mkdir(parents=True)
            (external / "bin/python").write_text("external", encoding="utf-8")
            try:
                (root / setup.VENV_NAME).symlink_to(external, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks unavailable: {exc}")
            config = root / setup.RUNTIME_CONFIG_FILENAME
            config.write_text("{}", encoding="utf-8")
            with patch.object(setup.deps, "state_matches") as state_matches:
                result = setup._reusable_environment(
                    fake_context(root), fake_plan(), root / "cache", {"lock": 1}
                )
            self.assertIsNone(result)
            self.assertFalse(config.exists())
            self.assertEqual(
                (external / "bin/python").read_text(encoding="utf-8"), "external"
            )
            state_matches.assert_not_called()

    def test_failed_promotion_does_not_restore_config_for_alias_venv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            external = root / "external-venv"
            external.mkdir()
            (external / "keep.txt").write_text("keep", encoding="utf-8")
            try:
                (root / setup.VENV_NAME).symlink_to(external, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks unavailable: {exc}")
            staging = root / setup.VENV_STAGING_NAME
            (staging / "bin").mkdir(parents=True)
            (staging / "bin/python").write_text("new", encoding="utf-8")
            state_staging = root / setup.STATE_STAGING_FILENAME
            state_staging.write_text("new-state", encoding="utf-8")
            config = root / setup.RUNTIME_CONFIG_FILENAME
            config.write_text("old-config", encoding="utf-8")
            with (
                patch.object(setup, "interpreter_fingerprint", return_value=FINGERPRINT),
                patch.object(setup.deps, "cpu_build_environment", return_value={}),
                patch.object(
                    setup,
                    "_run_checked",
                    side_effect=setup.SetupFailure("PROMOTED_FAIL", "simulated failure"),
                ),
            ):
                with self.assertRaises(setup.SetupFailure) as raised:
                    setup._promote_environment(
                        fake_context(root),
                        fake_plan(),
                        root / "cache",
                        staging,
                        state_staging,
                        {"ok": True},
                        {"ok": True},
                    )
            self.assertEqual(raised.exception.code, "PROMOTED_FAIL")
            self.assertFalse((root / setup.VENV_NAME).exists())
            self.assertFalse(config.exists())
            self.assertFalse((root / setup.CONFIG_BACKUP_FILENAME).exists())
            self.assertEqual((external / "keep.txt").read_text(encoding="utf-8"), "keep")

    def test_venv_and_checked_commands_receive_sanitized_environment(self) -> None:
        inherited = {
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": "/tmp/injected",
            "PIP_CONFIG_FILE": "/tmp/evil.conf",
            "SERVICE_SECRET": "secret",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            context = fake_context(root)
            target = root / setup.VENV_STAGING_NAME

            def create_venv(_command, **_kwargs):
                python = target / "bin/python"
                python.parent.mkdir(parents=True, exist_ok=True)
                python.touch()
                return types.SimpleNamespace()

            with patch.dict(setup.os.environ, inherited, clear=True), patch.object(
                setup.subprocess, "run", side_effect=create_venv
            ) as venv_run, patch.object(
                setup, "interpreter_fingerprint", return_value=FINGERPRINT
            ):
                setup._create_venv(context, target)
            venv_env = venv_run.call_args.kwargs["env"]

            with patch.dict(setup.os.environ, inherited, clear=True), patch.object(
                setup.subprocess, "run", return_value=types.SimpleNamespace()
            ) as checked_run:
                setup._run_checked([sys.executable, "-c", "pass"], stage="probe")
            checked_env = checked_run.call_args.kwargs["env"]

            with patch.dict(setup.os.environ, inherited, clear=True), patch.object(
                setup.subprocess, "run", return_value=types.SimpleNamespace()
            ) as pip_run:
                setup._run_checked(
                    [sys.executable, "-m", "pip", "--isolated", "check"],
                    stage="pip probe",
                )
            pip_env = pip_run.call_args.kwargs["env"]

        for child_env in (venv_env, checked_env):
            self.assertEqual(child_env, {"PATH": "/usr/bin:/bin"})
        self.assertEqual(pip_env["PATH"], "/usr/bin:/bin")
        self.assertEqual(pip_env["PIP_CONFIG_FILE"], setup.os.devnull)
        self.assertNotIn("SERVICE_SECRET", pip_env)

    def test_venv_python_uses_native_windows_and_posix_layouts(self) -> None:
        root = Path("/extension/venv")
        self.assertEqual(setup.venv_python(root, "linux"), root / "bin/python")
        self.assertEqual(
            setup.venv_python(root, "win32"), root / "Scripts/python.exe"
        )

    def test_directory_reparse_venv_is_removed_without_recursing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            venv = root / "venv"
            venv.mkdir()
            with (
                patch.object(setup, "_is_alias", return_value=True),
                patch.object(setup.shutil, "rmtree") as recursive_remove,
            ):
                setup._remove_venv(venv, root)
            self.assertFalse(venv.exists())
            recursive_remove.assert_not_called()

    def test_repair_reuses_only_fingerprinted_and_smoked_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python = root / "venv/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            context = fake_context(root)
            plan = fake_plan()
            with (
                patch.object(setup.deps, "dependency_state_payload", return_value={"lock": 1}),
                patch.object(setup.deps, "state_matches", return_value=True),
                patch.object(setup, "interpreter_fingerprint", return_value=FINGERPRINT),
                patch.object(setup.deps, "verify_dependencies", return_value={"ok": True}),
                patch.object(
                    setup.deps,
                    "verify_portable_cpu_extension",
                    return_value={"ok": True, "operator": "tetrahedron"},
                ),
                patch.object(setup, "_invalidate_runtime_config") as invalidate,
                patch.object(setup, "_install_environment") as install,
                patch.object(setup, "_require_free_space") as space,
            ):
                result = setup.install_or_reuse_environment(
                    context, plan, root / "cache", types.SimpleNamespace()
                )
            self.assertTrue(result.reused)
            invalidate.assert_not_called()
            install.assert_not_called()
            space.assert_not_called()

    def test_runtime_config_is_invalidated_when_existing_venv_fails_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python = root / "venv/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            config = root / setup.RUNTIME_CONFIG_FILENAME
            config.write_text("{}", encoding="utf-8")
            context = fake_context(root)
            plan = fake_plan()
            wrong = {**FINGERPRINT, "cache_tag": "different"}
            with (
                patch.object(setup.deps, "state_matches", return_value=True),
                patch.object(setup, "interpreter_fingerprint", return_value=wrong),
            ):
                result = setup._reusable_environment(
                    context, plan, root / "cache", {"lock": 1}
                )
            self.assertIsNone(result)
            self.assertFalse(config.exists())

    def test_fresh_install_writes_state_only_after_all_smokes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "cache").mkdir()
            context = fake_context(root)
            plan = fake_plan()
            events: list[str] = []
            checked_commands: list[list[str]] = []
            cpu_build = types.SimpleNamespace(
                pip_install_args=(
                    "-m",
                    "pip",
                    "install",
                    "--no-build-isolation",
                    "--no-deps",
                    "/build",
                )
            )

            def make_python(_context, target_venv):
                python = target_venv / "bin/python"
                python.parent.mkdir(parents=True, exist_ok=True)
                python.touch()
                events.append("venv")
                return python

            def write_state(path, payload):
                path.write_text(json.dumps(payload), encoding="utf-8")
                events.append("state-staging")

            def run_checked(command, **kwargs):
                checked_commands.append(list(command))
                events.append(
                    "pip-check-promoted"
                    if kwargs["stage"] == "Checking the promoted dependency graph"
                    else "pip-check-staging"
                    if command[-1:] == ["check"] and "pip" in command
                    else "cpu-install"
                )

            with (
                patch.object(setup.deps, "dependency_state_payload", return_value={"lock": 1}),
                patch.object(setup, "_reusable_environment", return_value=None),
                patch.object(setup, "_require_free_space"),
                patch.object(setup, "_create_venv", side_effect=make_python),
                patch.object(setup, "interpreter_fingerprint", return_value=FINGERPRINT),
                patch.object(
                    setup.deps,
                    "install_dependencies",
                    side_effect=lambda *_a, **_k: events.append("dependencies")
                    or root / "cache/dependency-constraints/locked.txt",
                ),
                patch.object(
                    setup.deps,
                    "validate_dependency_constraints_file",
                    side_effect=lambda path, _plan: path,
                ),
                patch.object(setup.deps, "cpu_build_environment", return_value={}),
                patch.object(
                    setup.deps,
                    "prepare_build_workspace",
                    return_value=root / "cpu-workspace",
                ),
                patch.object(
                    setup,
                    "_run_checked",
                    side_effect=run_checked,
                ),
                patch.object(
                    setup.deps,
                    "verify_dependencies",
                    side_effect=lambda *_a, **_k: events.append("dependency-smoke") or {"ok": True},
                ),
                patch.object(
                    setup.deps,
                    "verify_portable_cpu_extension",
                    side_effect=lambda *_a, **_k: events.append("cpu-smoke") or {"ok": True},
                ),
                patch.object(
                    setup.deps,
                    "write_state",
                    side_effect=write_state,
                ),
            ):
                result = setup.install_or_reuse_environment(context, plan, root / "cache", cpu_build)

            self.assertFalse(result.reused)
            self.assertEqual(result.python, root / "venv/bin/python")
            self.assertTrue(result.python.is_file())
            self.assertFalse((root / setup.VENV_STAGING_NAME).exists())
            self.assertFalse((root / setup.VENV_BACKUP_NAME).exists())
            self.assertEqual(
                events,
                [
                    "venv",
                    "dependencies",
                    "cpu-install",
                    "pip-check-staging",
                    "dependency-smoke",
                    "cpu-smoke",
                    "state-staging",
                    "pip-check-promoted",
                    "dependency-smoke",
                    "cpu-smoke",
                ],
            )
            self.assertEqual(len(checked_commands), 3)
            self.assertTrue(
                all("--isolated" in command for command in checked_commands)
            )
            self.assertIn("--no-index", checked_commands[0])
            self.assertIn("--constraint", checked_commands[0])
            constraint_position = checked_commands[0].index("--constraint")
            self.assertEqual(
                checked_commands[0][constraint_position + 1],
                str(root / "cache/dependency-constraints/locked.txt"),
            )
            self.assertEqual(checked_commands[0][-1], str(root / "cpu-workspace"))

    def test_failed_staging_rebuild_preserves_previous_installation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            old_python = root / "venv/bin/python"
            old_python.parent.mkdir(parents=True)
            old_python.write_text("old", encoding="utf-8")
            config = root / setup.RUNTIME_CONFIG_FILENAME
            state = root / setup.SETUP_STATE_FILENAME
            config.write_text('{"old": true}', encoding="utf-8")
            state.write_text('{"old": true}', encoding="utf-8")
            context = fake_context(root)
            plan = fake_plan()
            with (
                patch.object(setup.deps, "dependency_state_payload", return_value={"lock": 1}),
                patch.object(setup, "_reusable_environment", return_value=None),
                patch.object(
                    setup,
                    "_install_environment",
                    side_effect=setup.SetupFailure("INSTALL_FAILED", "simulated failure"),
                ),
            ):
                with self.assertRaises(setup.SetupFailure):
                    setup.install_or_reuse_environment(
                        context, plan, root / "cache", types.SimpleNamespace()
                    )
            self.assertEqual(old_python.read_text(encoding="utf-8"), "old")
            self.assertEqual(config.read_text(encoding="utf-8"), '{"old": true}')
            self.assertEqual(state.read_text(encoding="utf-8"), '{"old": true}')

    def test_partial_staging_creation_is_cleaned_without_touching_old_venv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            old_python = root / "venv/bin/python"
            old_python.parent.mkdir(parents=True)
            old_python.write_text("old", encoding="utf-8")
            context = fake_context(root)
            plan = fake_plan()

            def fail_after_partial_create(_context, target_venv):
                target_venv.mkdir()
                (target_venv / "partial").write_text("partial", encoding="utf-8")
                raise setup.SetupFailure("VENV_CREATE_FAILED", "simulated failure")

            with (
                patch.object(setup.deps, "dependency_state_payload", return_value={"lock": 1}),
                patch.object(setup, "_reusable_environment", return_value=None),
                patch.object(setup, "_require_free_space"),
                patch.object(setup, "_create_venv", side_effect=fail_after_partial_create),
            ):
                with self.assertRaises(setup.SetupFailure):
                    setup.install_or_reuse_environment(
                        context, plan, root / "cache", types.SimpleNamespace()
                    )
            self.assertEqual(old_python.read_text(encoding="utf-8"), "old")
            self.assertFalse((root / setup.VENV_STAGING_NAME).exists())

    def test_failed_post_promotion_smoke_restores_venv_state_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "cache").mkdir()
            old_python = root / "venv/bin/python"
            old_python.parent.mkdir(parents=True)
            old_python.write_text("old-python", encoding="utf-8")
            config = root / setup.RUNTIME_CONFIG_FILENAME
            state = root / setup.SETUP_STATE_FILENAME
            config.write_text('{"generation": "old"}', encoding="utf-8")
            state.write_text('{"generation": "old"}', encoding="utf-8")
            context = fake_context(root)
            plan = fake_plan()
            cpu_build = types.SimpleNamespace(
                pip_install_args=(
                    "-m",
                    "pip",
                    "install",
                    "--no-build-isolation",
                    "--no-deps",
                    "/build",
                )
            )
            promoted_config_visibility: list[bool] = []
            dependency_checks = 0

            def make_staging(_context, target_venv):
                python = target_venv / "bin/python"
                python.parent.mkdir(parents=True)
                python.write_text("new-python", encoding="utf-8")
                return python

            def write_staging_state(path, payload):
                path.write_text(json.dumps(payload), encoding="utf-8")

            def checked(_command, *, stage, **_kwargs):
                if stage == "Checking the promoted dependency graph":
                    promoted_config_visibility.append(config.exists())

            def dependency_smoke(*_args, **_kwargs):
                nonlocal dependency_checks
                dependency_checks += 1
                if dependency_checks == 2:
                    raise setup.deps.DependencyError(
                        "POST_PROMOTION_SMOKE_FAILED", "simulated relocated-venv failure"
                    )
                return {"ok": True}

            with (
                patch.object(setup.deps, "dependency_state_payload", return_value={"lock": 1}),
                patch.object(setup, "_reusable_environment", return_value=None),
                patch.object(setup, "_require_free_space"),
                patch.object(setup, "_create_venv", side_effect=make_staging),
                patch.object(setup, "interpreter_fingerprint", return_value=FINGERPRINT),
                patch.object(
                    setup.deps,
                    "install_dependencies",
                    return_value=root / "cache/dependency-constraints/locked.txt",
                ),
                patch.object(
                    setup.deps,
                    "validate_dependency_constraints_file",
                    side_effect=lambda path, _plan: path,
                ),
                patch.object(setup.deps, "cpu_build_environment", return_value={}),
                patch.object(
                    setup.deps,
                    "prepare_build_workspace",
                    return_value=root / "cpu-workspace",
                ),
                patch.object(setup, "_run_checked", side_effect=checked),
                patch.object(setup.deps, "verify_dependencies", side_effect=dependency_smoke),
                patch.object(
                    setup.deps,
                    "verify_portable_cpu_extension",
                    return_value={"ok": True},
                ),
                patch.object(setup.deps, "write_state", side_effect=write_staging_state),
            ):
                with self.assertRaises(setup.SetupFailure) as raised:
                    setup.install_or_reuse_environment(
                        context, plan, root / "cache", cpu_build
                    )

            self.assertEqual(raised.exception.code, "VENV_PROMOTION_FAILED")
            self.assertEqual(promoted_config_visibility, [False])
            self.assertEqual(old_python.read_text(encoding="utf-8"), "old-python")
            self.assertEqual(config.read_text(encoding="utf-8"), '{"generation": "old"}')
            self.assertEqual(state.read_text(encoding="utf-8"), '{"generation": "old"}')
            self.assertFalse((root / setup.VENV_STAGING_NAME).exists())
            self.assertFalse((root / setup.VENV_BACKUP_NAME).exists())
            self.assertFalse((root / setup.CONFIG_BACKUP_FILENAME).exists())
            self.assertFalse((root / setup.STATE_STAGING_FILENAME).exists())

    def test_interrupted_promotion_recovers_previous_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            current = root / setup.VENV_NAME
            backup = root / setup.VENV_BACKUP_NAME
            (current / "bin").mkdir(parents=True)
            (backup / "bin").mkdir(parents=True)
            (current / "bin/python").write_text("new", encoding="utf-8")
            (backup / "bin/python").write_text("old", encoding="utf-8")
            (root / setup.CONFIG_BACKUP_FILENAME).write_text("old-config", encoding="utf-8")
            (root / setup.STATE_BACKUP_FILENAME).write_text("old-state", encoding="utf-8")
            (root / setup.SETUP_STATE_FILENAME).write_text("new-state", encoding="utf-8")
            setup._recover_environment_transaction(fake_context(root))
            self.assertEqual((root / "venv/bin/python").read_text(encoding="utf-8"), "old")
            self.assertEqual(
                (root / setup.RUNTIME_CONFIG_FILENAME).read_text(encoding="utf-8"),
                "old-config",
            )
            self.assertEqual(
                (root / setup.SETUP_STATE_FILENAME).read_text(encoding="utf-8"),
                "old-state",
            )
            self.assertFalse(backup.exists())


class IntegrationTests(unittest.TestCase):
    def _run_setup_patches(self, root: Path, *, plan) -> tuple[object, dict, list[str]]:
        models = root / "models"
        revision = models / setup.EXTENSION_ID / "lato2/revisions" / setup.REVISION_ID
        cache = revision / "runtime-cache"
        source = revision / "source/LATO.2"
        for directory in (models, revision, cache, source):
            directory.mkdir(parents=True, exist_ok=True)
        context = fake_context(root)
        portable_report = types.SimpleNamespace(to_dict=lambda: {"reused": True})
        cpu_sources = types.SimpleNamespace(ovoxel=cache / "ovo", eigen=cache / "eigen")
        cpu_build = types.SimpleNamespace(
            pip_install_args=(
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "--no-deps",
                "/build",
            ),
            to_dict=lambda: {"reused": True},
        )
        environment = setup.EnvironmentResult(
            root / "venv/bin/python", True, {"ok": True}, {"ok": True}
        )
        written: dict = {}
        events: list[str] = []

        def write_config(ext, model_root, revision_root, *, extra):
            events.append("config")
            written.update(extra)
            self.assertEqual(ext, root)
            self.assertEqual(model_root, models)
            self.assertEqual(revision_root, revision)
            return root / "runtime_config.json"

        patches = (
            patch.object(setup, "validate_context", return_value=context),
            patch.object(setup, "resolve_models_root", return_value=models),
            patch.object(setup, "owned_snapshot_directory", return_value=revision),
            patch.object(setup, "safe_snapshot_directory", return_value=cache),
            patch.object(setup.deps, "select_dependency_plan", return_value=plan),
            patch.object(setup, "_preflight_plan", side_effect=lambda *_: events.append("plan")),
            patch.object(setup, "_preflight_assets", side_effect=lambda *_: events.append("assets-space")),
            patch.object(setup, "ensure_snapshot", side_effect=lambda *_a, **_k: events.append("snapshot") or revision),
            patch.object(setup, "verify_snapshot", return_value=[]),
            patch.object(setup, "materialize_portable_runtime", return_value=portable_report),
            patch.object(setup.deps, "prepare_portable_cpu_sources", return_value=cpu_sources),
            patch.object(setup, "materialize_ovoxel_cpu_build", return_value=cpu_build),
            patch.object(setup, "install_or_reuse_environment", side_effect=lambda *_: events.append("environment") or environment),
            patch.object(setup.deps, "dependency_lock_digest", return_value="a" * 64),
            patch.object(setup, "write_runtime_config", side_effect=write_config),
        )
        return patches, written, events

    def test_portable_publishes_only_the_backend_that_passed_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            patches, written, events = self._run_setup_patches(root, plan=fake_plan())
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], patches[10], patches[11], patches[12], patches[13], patches[14]:
                result = setup.run_setup({}, root)
            self.assertEqual(result, root / "runtime_config.json")
            self.assertEqual(written["available_backends"], ["portable"])
            self.assertEqual(written["default_backend"], "portable")
            self.assertEqual(events[-1], "config")

    def test_exact_publishes_upstream_and_additive_portable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            patches, written, _events = self._run_setup_patches(root, plan=fake_plan(exact=True))
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], patches[10], patches[11], patches[12], patches[13], patches[14]:
                setup.run_setup({}, root)
            self.assertEqual(written["available_backends"], ["upstream", "portable"])
            self.assertEqual(written["default_backend"], "upstream")

    def test_runtime_config_failure_restores_previous_repair_generation_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            patches, _written, _events = self._run_setup_patches(
                root, plan=fake_plan()
            )
            patches = list(patches)

            current_venv = root / setup.VENV_NAME
            backup_venv = root / setup.VENV_BACKUP_NAME
            (current_venv / "bin").mkdir(parents=True)
            (backup_venv / "bin").mkdir(parents=True)
            (current_venv / "bin/python").write_bytes(b"new-python")
            (current_venv / "new-only").write_bytes(b"new-only")
            (backup_venv / "bin/python").write_bytes(b"old-python\x00exact")
            (backup_venv / "old-only").write_bytes(b"old-only")
            (root / setup.SETUP_STATE_FILENAME).write_bytes(b"new-state")
            (root / setup.STATE_BACKUP_FILENAME).write_bytes(
                b"old-state\x00exact"
            )
            (root / setup.CONFIG_BACKUP_FILENAME).write_bytes(
                b"old-config\x00exact"
            )
            environment = setup.EnvironmentResult(
                current_venv / "bin/python",
                False,
                {"ok": True},
                {"ok": True},
                setup.EnvironmentPromotion(True, True, True),
            )
            patches[12] = patch.object(
                setup, "install_or_reuse_environment", return_value=environment
            )

            def fail_config(*_args, **_kwargs):
                # The prior generation must still be recoverable at the exact
                # point where publication begins.
                self.assertTrue(backup_venv.is_dir())
                self.assertTrue((root / setup.STATE_BACKUP_FILENAME).is_file())
                self.assertTrue((root / setup.CONFIG_BACKUP_FILENAME).is_file())
                self.assertFalse((root / setup.RUNTIME_CONFIG_FILENAME).exists())
                (root / setup.RUNTIME_CONFIG_FILENAME).write_bytes(b"new-config")
                raise OSError("simulated runtime_config publication failure")

            patches[14] = patch.object(
                setup, "write_runtime_config", side_effect=fail_config
            )
            with ExitStack() as stack:
                for setup_patch in patches:
                    stack.enter_context(setup_patch)
                with self.assertRaises(OSError):
                    setup.run_setup({}, root)

            self.assertEqual(
                (root / setup.VENV_NAME / "bin/python").read_bytes(),
                b"old-python\x00exact",
            )
            self.assertTrue((root / setup.VENV_NAME / "old-only").is_file())
            self.assertFalse((root / setup.VENV_NAME / "new-only").exists())
            self.assertEqual(
                (root / setup.SETUP_STATE_FILENAME).read_bytes(),
                b"old-state\x00exact",
            )
            self.assertEqual(
                (root / setup.RUNTIME_CONFIG_FILENAME).read_bytes(),
                b"old-config\x00exact",
            )
            for transaction_path in (
                setup.VENV_STAGING_NAME,
                setup.VENV_BACKUP_NAME,
                setup.STATE_STAGING_FILENAME,
                setup.STATE_BACKUP_FILENAME,
                setup.CONFIG_BACKUP_FILENAME,
            ):
                self.assertFalse((root / transaction_path).exists())

    def test_first_install_config_failure_keeps_verified_state_without_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            patches, _written, _events = self._run_setup_patches(
                root, plan=fake_plan()
            )
            patches = list(patches)
            current_venv = root / setup.VENV_NAME
            (current_venv / "bin").mkdir(parents=True)
            (current_venv / "bin/python").write_bytes(b"verified-python")
            (root / setup.SETUP_STATE_FILENAME).write_bytes(b"verified-state")
            environment = setup.EnvironmentResult(
                current_venv / "bin/python",
                False,
                {"ok": True},
                {"ok": True},
                setup.EnvironmentPromotion(False, False, False),
            )
            patches[12] = patch.object(
                setup, "install_or_reuse_environment", return_value=environment
            )

            def fail_config(*_args, **_kwargs):
                (root / setup.RUNTIME_CONFIG_FILENAME).write_bytes(b"partial-config")
                raise OSError("simulated runtime_config publication failure")

            patches[14] = patch.object(
                setup, "write_runtime_config", side_effect=fail_config
            )
            with ExitStack() as stack:
                for setup_patch in patches:
                    stack.enter_context(setup_patch)
                with self.assertRaises(OSError):
                    setup.run_setup({}, root)

            self.assertEqual(
                (current_venv / "bin/python").read_bytes(), b"verified-python"
            )
            self.assertEqual(
                (root / setup.SETUP_STATE_FILENAME).read_bytes(), b"verified-state"
            )
            self.assertFalse((root / setup.RUNTIME_CONFIG_FILENAME).exists())
            self.assertFalse((root / setup.VENV_BACKUP_NAME).exists())
            self.assertFalse((root / setup.STATE_BACKUP_FILENAME).exists())
            self.assertFalse((root / setup.CONFIG_BACKUP_FILENAME).exists())

    def test_preflight_failure_never_switches_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            models = root / "models"
            revision = models / setup.EXTENSION_ID / "lato2/revisions" / setup.REVISION_ID
            cache = revision / "runtime-cache"
            for directory in (models, revision, cache):
                directory.mkdir(parents=True, exist_ok=True)
            context = fake_context(root)
            plan = fake_plan(exact=True)
            error = setup.deps.DependencyError("NVCC_MISSING", "CUDA Toolkit 12.4 is required")
            with (
                patch.object(setup, "validate_context", return_value=context),
                patch.object(setup, "resolve_models_root", return_value=models) as resolver,
                patch.object(setup, "owned_snapshot_directory", return_value=revision),
                patch.object(setup, "safe_snapshot_directory", return_value=cache),
                patch.object(setup.deps, "select_dependency_plan", return_value=plan) as select,
                patch.object(setup, "_preflight_plan", side_effect=error),
                patch.object(setup, "ensure_snapshot") as snapshot,
                patch.object(setup, "write_runtime_config") as config,
            ):
                with self.assertRaises(setup.deps.DependencyError):
                    setup.run_setup({}, root)
            select.assert_called_once()
            resolver.assert_called_once_with(
                context.payload,
                context.ext_dir,
                context.platform_name,
                payload_keys=setup.SETUP_MODELS_PAYLOAD_KEYS,
                require_existing=True,
            )
            snapshot.assert_not_called()
            config.assert_not_called()

    def test_run_setup_holds_the_extension_lock_around_all_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            context = fake_context(root)
            events: list[str] = []

            class LockProbe:
                def __enter__(self):
                    events.append("lock-enter")

                def __exit__(self, *_args):
                    events.append("lock-exit")

            with (
                patch.object(setup, "validate_context", return_value=context),
                patch.object(setup, "setup_lock", return_value=LockProbe()) as lock,
                patch.object(
                    setup,
                    "_run_setup_locked",
                    side_effect=lambda value: events.append("mutations")
                    or root / setup.RUNTIME_CONFIG_FILENAME,
                ),
            ):
                result = setup.run_setup({}, root)
            self.assertEqual(result, root / setup.RUNTIME_CONFIG_FILENAME)
            self.assertEqual(events, ["lock-enter", "mutations", "lock-exit"])
            lock.assert_called_once_with(root, platform_name="linux")


if __name__ == "__main__":
    unittest.main()
