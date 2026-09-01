from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import stat
import tempfile
import unittest
from unittest import mock

from lato2_modly import dependencies as deps


def context(
    *,
    system: str = "linux",
    arch: str = "x86_64",
    sm: int = 89,
    cuda: int = 124,
) -> dict[str, object]:
    return {
        "platform": system,
        "arch": arch,
        "gpu_sm": sm,
        "cuda_version": cuda,
        "accelerator": "cuda",
    }


class PlanSelectionTests(unittest.TestCase):
    def test_linux_ampere_auto_is_complete_exact_profile(self) -> None:
        plan = deps.select_dependency_plan(context(sm=86))
        self.assertEqual(plan.profile, "exact-upstream")
        self.assertEqual(plan.torch_lane, "cu124")
        self.assertEqual(plan.attention_backend, "flash_attn")
        self.assertTrue(plan.install_flash_attn)
        self.assertTrue(plan.install_native_stack)

    def test_windows_defaults_to_portable_without_native_toolchain(self) -> None:
        plan = deps.select_dependency_plan(
            context(system="win32", arch="AMD64", sm=75)
        )
        self.assertEqual(plan.profile, "portable")
        self.assertEqual(plan.attention_backend, "sdpa")
        self.assertFalse(plan.install_native_stack)
        self.assertIn("open3d", deps.expected_distribution_versions(plan))

    def test_linux_turing_defaults_to_portable_and_exact_is_rejected(self) -> None:
        payload = context(sm=75)
        self.assertEqual(deps.select_dependency_plan(payload).profile, "portable")
        with self.assertRaises(deps.DependencyError) as raised:
            deps.select_dependency_plan(payload, "exact-upstream")
        self.assertEqual(raised.exception.code, "EXACT_BF16_UNSUPPORTED")

    def test_arm64_sm90_uses_cu126_and_unsuffixed_torchvision(self) -> None:
        plan = deps.select_dependency_plan(
            context(arch="aarch64", sm=90, cuda=126)
        )
        self.assertEqual(plan.profile, "portable")
        self.assertEqual(plan.torch_requirements[0], "torch==2.6.0+cu126")
        self.assertEqual(plan.torch_requirements[1], "torchvision==0.21.0")
        self.assertNotIn("open3d", deps.expected_distribution_versions(plan))

    def test_arm64_sm120_uses_cu128_and_unsuffixed_torchvision(self) -> None:
        plan = deps.select_dependency_plan(
            context(arch="arm64", sm=120, cuda=128)
        )
        self.assertEqual(plan.torch_lane, "cu128")
        self.assertEqual(plan.torch_requirements[0], "torch==2.9.1+cu128")
        self.assertEqual(plan.torch_requirements[1], "torchvision==0.24.1")
        self.assertEqual(plan.support_level, "experimental")

    def test_x64_torchvision_keeps_cuda_suffix(self) -> None:
        plan = deps.select_dependency_plan(
            context(system="win32", arch="x64", sm=120, cuda=128)
        )
        self.assertEqual(plan.torch_requirements[1], "torchvision==0.24.1+cu128")

    def test_sm120_rejects_driver_metadata_older_than_cu128(self) -> None:
        with self.assertRaises(deps.DependencyError) as raised:
            deps.select_dependency_plan(
                context(system="linux", arch="x64", sm=120, cuda=126)
            )
        self.assertEqual(raised.exception.code, "DRIVER_CUDA_TOO_OLD")

    def test_old_driver_fails_before_install(self) -> None:
        with self.assertRaises(deps.DependencyError) as raised:
            deps.select_dependency_plan(context(sm=86, cuda=120))
        self.assertEqual(raised.exception.code, "DRIVER_CUDA_TOO_OLD")

    def test_exact_arm64_fails_closed(self) -> None:
        with self.assertRaises(deps.DependencyError) as raised:
            deps.select_dependency_plan(
                context(arch="arm64", sm=90, cuda=126), "exact-upstream"
            )
        self.assertEqual(raised.exception.code, "EXACT_ARM64_UNAVAILABLE")

    def test_cpu_is_never_silently_selected(self) -> None:
        payload = context()
        payload.update(accelerator="cpu", gpu_sm=0)
        with self.assertRaises(deps.DependencyError) as raised:
            deps.select_dependency_plan(payload)
        self.assertEqual(raised.exception.code, "CUDA_REQUIRED")


class LockAndStateTests(unittest.TestCase):
    def test_complete_constraints_are_platform_and_torch_lane_specific(self) -> None:
        linux_cu124 = deps.select_dependency_plan(context(sm=75), "portable")
        windows_cu124 = deps.select_dependency_plan(
            context(system="win32", arch="x64", sm=75), "portable"
        )
        windows_cu124_exact = deps.select_dependency_plan(
            context(system="win32", arch="x64", sm=86), "exact-upstream"
        )
        arm_cu126 = deps.select_dependency_plan(
            context(arch="arm64", sm=90, cuda=126), "portable"
        )
        linux_cu128 = deps.select_dependency_plan(
            context(arch="x64", sm=120, cuda=128), "portable"
        )
        windows_cu128 = deps.select_dependency_plan(
            context(system="win32", arch="x64", sm=120, cuda=128), "portable"
        )
        arm_cu128 = deps.select_dependency_plan(
            context(arch="arm64", sm=120, cuda=128), "portable"
        )

        expected_counts = {
            linux_cu124: 110,
            windows_cu124: 80,
            windows_cu124_exact: 96,
            arm_cu126: 34,
            linux_cu128: 112,
            windows_cu128: 80,
            arm_cu128: 50,
        }
        for plan, count in expected_counts.items():
            locked = deps.constraint_requirements(plan)
            self.assertEqual(len(locked), count)
            self.assertEqual(
                deps.dependency_lock_payload(plan)["constraints"], list(locked)
            )
            normalized = [deps._requirement_name(item) for item in locked]
            self.assertEqual(len(normalized), len(set(normalized)))
            for direct in (
                *deps.BOOTSTRAP_REQUIREMENTS,
                *deps.BASE_REQUIREMENTS,
                *plan.torch_requirements,
            ):
                self.assertIn(direct, locked)

        linux_124 = deps.expected_distribution_versions(linux_cu124)
        windows_124 = deps.expected_distribution_versions(windows_cu124)
        arm_126 = deps.expected_distribution_versions(arm_cu126)
        linux_128 = deps.expected_distribution_versions(linux_cu128)
        windows_128 = deps.expected_distribution_versions(windows_cu128)
        arm_128 = deps.expected_distribution_versions(arm_cu128)
        self.assertEqual(linux_124["sympy"], "1.13.1")
        self.assertEqual(arm_126["sympy"], "1.13.1")
        self.assertEqual(linux_128["sympy"], "1.14.0")
        self.assertEqual(arm_128["sympy"], "1.14.0")
        self.assertIn("triton", linux_124)
        self.assertNotIn("triton", windows_124)
        self.assertNotIn("triton", arm_126)
        self.assertIn("triton", linux_128)
        self.assertNotIn("triton", windows_128)
        self.assertIn("triton", arm_128)
        self.assertIn("matplotlib", linux_124)
        self.assertNotIn("matplotlib", windows_124)
        self.assertEqual(windows_124["colorama"], "0.4.6")
        self.assertNotIn("open3d", arm_126)
        self.assertEqual(len({deps.dependency_lock_digest(p) for p in expected_counts}), 7)

    def test_complete_constraints_match_recursive_official_metadata_snapshots(self) -> None:
        # Frozen from recursive uv CPython 3.11 resolutions against PyPI,
        # download.pytorch.org and the exact PyG wheel page.  Hashing every
        # sorted requirement catches an omission that a package count cannot.
        plans = {
            "linux-x64-cu124-portable": deps.select_dependency_plan(
                context(sm=75), "portable"
            ),
            "linux-x64-cu124-exact": deps.select_dependency_plan(
                context(sm=86), "exact-upstream"
            ),
            "win32-x64-cu124-portable": deps.select_dependency_plan(
                context(system="win32", arch="x64", sm=75), "portable"
            ),
            "win32-x64-cu124-exact": deps.select_dependency_plan(
                context(system="win32", arch="x64", sm=86), "exact-upstream"
            ),
            "linux-arm64-cu126": deps.select_dependency_plan(
                context(arch="arm64", sm=90, cuda=126), "portable"
            ),
            "linux-x64-cu128": deps.select_dependency_plan(
                context(arch="x64", sm=120, cuda=128), "portable"
            ),
            "win32-x64-cu128": deps.select_dependency_plan(
                context(system="win32", arch="x64", sm=120, cuda=128),
                "portable",
            ),
            "linux-arm64-cu128": deps.select_dependency_plan(
                context(arch="arm64", sm=120, cuda=128), "portable"
            ),
        }
        expected = {
            "linux-x64-cu124-portable": "15a0e9f6a237d7248d335ad016012a720cd527be51fe5ca182c17bad5741ca40",
            "linux-x64-cu124-exact": "ca1053f2ca5856caa0c4910d9423dfe652c47402c8db1beb427e8995107d6923",
            "win32-x64-cu124-portable": "297bedec9332cf2f5f63241e5f7d9d9d6a62b3ea95e07f7d181ddeaae485b129",
            "win32-x64-cu124-exact": "d4b7573a301797e76b690d2c3d97ff123ebd529f382d8beca14f64adf77048c2",
            "linux-arm64-cu126": "9e76edb3b0b594869bda25ea6f743dac5b3c6fa08136568cc84744f53f163165",
            "linux-x64-cu128": "4fb1d46987a126c04cd93b620cf7a4a4128a2fbfd7af86e3cf674faccbbdf849",
            "win32-x64-cu128": "2d40eb733066ba82b57966c275c0b811ca9714de1c0a0f5ff69145b9dc4ff16b",
            "linux-arm64-cu128": "89b6129130207920f4e8a4ea49c9c0971eafff2c56d0fc55a4776028dee24629",
        }
        for name, plan in plans.items():
            serialized = (
                "\n".join(sorted(deps.constraint_requirements(plan))) + "\n"
            ).encode("utf-8")
            self.assertEqual(hashlib.sha256(serialized).hexdigest(), expected[name])

    def test_exact_constraints_include_native_metadata_and_sparse_closure(self) -> None:
        plan = deps.select_dependency_plan(context(sm=86), "exact-upstream")
        versions = deps.expected_distribution_versions(plan)
        constrained_names = {
            deps._requirement_name(item)
            for item in deps.constraint_requirements(plan)
        }
        self.assertEqual(
            set(versions),
            constrained_names - {deps.OVOXEL_CPU_DISTRIBUTION},
        )
        for name, version in {
            "nvdiffrast": "0.4.0",
            "cumesh": "0.0.1",
            "flex-gemm": "1.0.0",
            "o-voxel": "0.0.1",
            "ccimport": "0.4.4",
            "pccm": "0.4.16",
            "pybind11": "3.1.0",
        }.items():
            self.assertEqual(versions[name], version)

    def test_constraint_selector_rejects_an_unaudited_lane_combination(self) -> None:
        plan = deps.select_dependency_plan(context(sm=75), "portable")
        invalid = replace(
            plan,
            arch="arm64",
            torch_lane="cu124",
            torch_requirements=deps._torch_requirements("cu124", "arm64"),
        )
        with self.assertRaises(deps.DependencyError) as raised:
            deps.constraint_requirements(invalid)
        self.assertEqual(raised.exception.code, "DEPENDENCY_PLAN_UNSUPPORTED")
        incoherent_profile = replace(plan, profile="exact-upstream")
        with self.assertRaises(deps.DependencyError) as raised:
            deps.constraint_requirements(incoherent_profile)
        self.assertEqual(raised.exception.code, "DEPENDENCY_PLAN_INVALID")
        incoherent_flash = replace(
            plan, install_flash_attn=True, attention_backend="flash_attn"
        )
        with self.assertRaises(deps.DependencyError) as raised:
            deps.constraint_requirements(incoherent_flash)
        self.assertEqual(raised.exception.code, "DEPENDENCY_PLAN_INVALID")

    def test_constraint_file_replaces_alias_and_wrong_content_atomically(self) -> None:
        plan = deps.select_dependency_plan(context(sm=75), "portable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            outside = root / "outside"
            cache.mkdir()
            outside.mkdir()
            marker = outside / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            (cache / "dependency-constraints").symlink_to(
                outside, target_is_directory=True
            )
            constraint = deps.materialize_dependency_constraints(cache, plan)
            self.assertTrue(marker.is_file())
            self.assertFalse((cache / "dependency-constraints").is_symlink())
            expected = deps._constraint_file_content(plan)
            self.assertEqual(constraint.read_bytes(), expected)
            constraint.write_text("tampered\n", encoding="utf-8")
            rebuilt = deps.materialize_dependency_constraints(cache, plan)
            self.assertEqual(rebuilt, constraint)
            self.assertEqual(rebuilt.read_bytes(), expected)
            rebuilt.write_text("swapped between pip stages\n", encoding="utf-8")
            with self.assertRaises(deps.DependencyError) as raised:
                deps._pip(
                    Path(__file__),
                    ["install", "example==1.0"],
                    env={},
                    runner=lambda *_args: self.fail("runner consumed a bad lock"),
                    log=lambda _message: None,
                    stage="test",
                    constraint_file=rebuilt,
                    constraint_plan=plan,
                )
            self.assertEqual(
                raised.exception.code, "DEPENDENCY_CONSTRAINTS_INVALID"
            )

    def test_profiles_have_distinct_locks_and_both_lock_cpu_addon(self) -> None:
        exact = deps.select_dependency_plan(context(sm=86), "exact-upstream")
        portable = deps.select_dependency_plan(context(sm=86), "portable")
        self.assertNotEqual(
            deps.dependency_lock_digest(exact), deps.dependency_lock_digest(portable)
        )
        self.assertIn("portableCpu", deps.dependency_lock_payload(exact))
        self.assertIn("portableCpu", deps.dependency_lock_payload(portable))
        portable_cpu = deps.dependency_lock_payload(portable)["portableCpu"]
        self.assertEqual(portable_cpu["distribution"], deps.OVOXEL_CPU_DISTRIBUTION)
        self.assertEqual(portable_cpu["version"], deps.OVOXEL_CPU_VERSION)
        self.assertEqual(
            portable_cpu["buildIdentity"], deps.OVOXEL_CPU_BUILD_IDENTITY
        )
        self.assertEqual(
            portable_cpu["templateTreeSha256"], deps.TEMPLATE_TREE_SHA256
        )
        self.assertEqual(
            portable_cpu["sourceTreeSha256"],
            deps.PORTABLE_CPU_SOURCE_TREE_SHA256,
        )
        self.assertEqual(
            portable_cpu["licenseSha256"],
            {
                name: digest
                for name, (_relative, digest) in sorted(
                    deps.LICENSE_SOURCE_SPECS.items()
                )
            },
        )
        exact_payload = json.dumps(deps.dependency_lock_payload(exact), sort_keys=True)
        self.assertEqual(
            deps.dependency_lock_payload(exact)["exact"]["sourceTreeSha256"],
            deps.NATIVE_SOURCE_TREE_SHA256[exact.system],
        )
        for revision in (
            deps.NVDIFFRAST_REVISION,
            deps.CUMESH_REVISION,
            deps.FLEXGEMM_REVISION,
            deps.TRELLIS2_REVISION,
        ):
            self.assertIn(revision, exact_payload)

    def test_portable_cpu_template_and_license_identity_change_the_lock(self) -> None:
        plan = deps.select_dependency_plan(context(sm=86), "portable")
        original = deps.dependency_lock_digest(plan)
        with mock.patch.object(deps, "TEMPLATE_TREE_SHA256", "0" * 64):
            self.assertNotEqual(deps.dependency_lock_digest(plan), original)
        replacement_licenses = {
            **deps.LICENSE_SOURCE_SPECS,
            "synthetic.txt": ("LICENSES/synthetic.txt", "1" * 64),
        }
        with mock.patch.object(
            deps, "LICENSE_SOURCE_SPECS", replacement_licenses
        ):
            self.assertNotEqual(deps.dependency_lock_digest(plan), original)

    def test_state_is_exact_and_rejects_symlinks(self) -> None:
        plan = deps.select_dependency_plan(context(sm=86))
        payload = deps.dependency_state_payload(plan, {"cache_tag": "cpython-311"})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "state.json"
            deps.write_state(state, payload)
            self.assertTrue(deps.state_matches(state, payload))
            changed = dict(payload)
            changed["lockDigest"] = "0" * 64
            self.assertFalse(deps.state_matches(state, changed))
            alias = root / "alias.json"
            try:
                alias.symlink_to(state)
            except OSError:
                self.skipTest("symlinks unavailable")
            self.assertFalse(deps.state_matches(alias, payload))

    def test_state_rejects_hardlinks(self) -> None:
        plan = deps.select_dependency_plan(context(sm=86))
        payload = deps.dependency_state_payload(plan, {"cache_tag": "cpython-311"})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside.json"
            deps.write_state(outside, payload)
            linked = root / "state.json"
            try:
                os.link(outside, linked)
            except OSError as exc:
                self.skipTest(f"hardlinks unavailable: {exc}")
            self.assertFalse(deps.state_matches(linked, payload))

    def test_all_source_archives_are_immutable_and_hashed(self) -> None:
        for spec in deps.SOURCE_ARCHIVES:
            self.assertRegex(spec.sha256, r"^[0-9a-f]{64}$")
            self.assertGreater(spec.size, 1_000)
            self.assertNotIn("/main", spec.url)
            self.assertNotIn("/master", spec.url)

    def test_cache_removal_rejects_symlink_parent_without_deleting_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside"
            outside.mkdir()
            victim = outside / "victim.txt"
            victim.write_text("keep", encoding="utf-8")
            alias = root / "alias"
            try:
                alias.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks unavailable: {exc}")
            with self.assertRaises(deps.DependencyError) as raised:
                deps._remove_owned_entry(alias / "victim.txt", alias)
            self.assertEqual(raised.exception.code, "CACHE_REPAIR_ESCAPE")
            self.assertEqual(victim.read_text(encoding="utf-8"), "keep")

    def test_windows_directory_reparse_is_removed_with_rmdir_not_unlink(self) -> None:
        parent = mock.MagicMock(spec=Path)
        child = mock.MagicMock(spec=Path)
        child.parent = parent
        info = mock.Mock(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        child.lstat.return_value = info
        deps._remove_owned_node(child)
        child.rmdir.assert_called_once_with()
        child.unlink.assert_not_called()


class BuildWorkspaceTests(unittest.TestCase):
    def test_torch_26_to_29_sm_transition_never_reuses_abi_artifacts(self) -> None:
        old_plan = deps.select_dependency_plan(
            context(arch="aarch64", sm=90, cuda=126), "portable"
        )
        new_plan = deps.select_dependency_plan(
            context(arch="aarch64", sm=120, cuda=128), "portable"
        )
        self.assertEqual(old_plan.torch_requirements[0], "torch==2.6.0+cu126")
        self.assertEqual(new_plan.torch_requirements[0], "torch==2.9.1+cu128")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source"
            package = source / "package"
            package.mkdir(parents=True)
            (source / "setup.py").write_text("from setuptools import setup\n", encoding="utf-8")
            (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
            (source / "build").mkdir()
            (source / "build" / "old.o").write_bytes(b"torch-2.6-object")
            (source / "dist").mkdir()
            (source / "dist" / "old.whl").write_bytes(b"old-wheel")
            (source / "package.egg-info").mkdir()
            (source / "package.egg-info" / "PKG-INFO").write_text("old", encoding="utf-8")
            (package / "inplace.so").write_bytes(b"old-abi")
            cache = root / "cache"

            old_workspace = deps.prepare_build_workspace(
                source, cache, old_plan, "ovoxel-cpu"
            )
            (old_workspace / "build").mkdir()
            (old_workspace / "build" / "compiled.o").write_bytes(b"torch-2.6")
            (old_workspace / "package" / "compiled.so").write_bytes(b"torch-2.6")

            new_workspace = deps.prepare_build_workspace(
                source, cache, new_plan, "ovoxel-cpu"
            )
            self.assertNotEqual(old_workspace, new_workspace)
            self.assertEqual(
                old_workspace.parent.name,
                deps.dependency_lock_digest(old_plan)[: deps._BUILD_LOCK_PREFIX_LENGTH],
            )
            self.assertEqual(
                new_workspace.parent.name,
                deps.dependency_lock_digest(new_plan)[: deps._BUILD_LOCK_PREFIX_LENGTH],
            )
            self.assertTrue((new_workspace / "package" / "__init__.py").is_file())
            for stale in (
                new_workspace / "build",
                new_workspace / "dist",
                new_workspace / "package.egg-info",
                new_workspace / "package" / "inplace.so",
                new_workspace / "package" / "compiled.so",
            ):
                self.assertFalse(stale.exists(), stale)

            # A Repair in the same lane also starts clean instead of consuming
            # the previous failed/in-place build output.
            (new_workspace / "build").mkdir()
            (new_workspace / "build" / "retry.o").write_bytes(b"stale")
            reset_workspace = deps.prepare_build_workspace(
                source, cache, new_plan, "ovoxel-cpu"
            )
            self.assertEqual(reset_workspace, new_workspace)
            self.assertFalse((reset_workspace / "build").exists())

    def test_workspace_copy_rejects_source_alias_without_following_it(self) -> None:
        plan = deps.select_dependency_plan(context(sm=86), "portable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source"
            source.mkdir()
            external = root / "external.txt"
            external.write_text("do not copy", encoding="utf-8")
            alias = source / "alias.txt"
            try:
                alias.symlink_to(external)
            except OSError:
                self.skipTest("symlinks unavailable")
            with self.assertRaises(deps.DependencyError) as raised:
                deps.prepare_build_workspace(source, root / "cache", plan, "alias-test")
            self.assertEqual(raised.exception.code, "BUILD_SOURCE_ALIAS")
            self.assertEqual(external.read_text(encoding="utf-8"), "do not copy")

    def test_workspace_reset_unlinks_alias_without_deleting_its_target(self) -> None:
        plan = deps.select_dependency_plan(context(sm=86), "portable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source"
            source.mkdir()
            (source / "setup.py").write_text("# source\n", encoding="utf-8")
            workspace = deps.prepare_build_workspace(
                source, root / "cache", plan, "safe-cleanup"
            )
            external = root / "external"
            external.mkdir()
            sentinel = external / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            alias = workspace / "escaped-build"
            try:
                alias.symlink_to(external, target_is_directory=True)
            except OSError:
                self.skipTest("directory symlinks unavailable")

            reset = deps.prepare_build_workspace(
                source, root / "cache", plan, "safe-cleanup"
            )
            self.assertEqual(reset, workspace)
            self.assertFalse((reset / "escaped-build").exists())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_wrong_full_lock_marker_under_same_prefix_is_rebuilt(self) -> None:
        plan = deps.select_dependency_plan(context(sm=86), "portable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source"
            source.mkdir()
            (source / "setup.py").write_text("# source\n", encoding="utf-8")
            workspace = deps.prepare_build_workspace(
                source, root / "cache", plan, "marker-test"
            )
            marker = workspace.parent / deps._BUILD_LOCK_MARKER
            wrong = deps._build_lock_payload(plan)
            wrong["lockDigest"] = (
                deps.dependency_lock_digest(plan)[: deps._BUILD_LOCK_PREFIX_LENGTH]
                + "0" * (64 - deps._BUILD_LOCK_PREFIX_LENGTH)
            )
            deps.write_state(marker, wrong)
            stale = workspace / "stale.o"
            stale.write_bytes(b"wrong lock ABI")

            rebuilt = deps.prepare_build_workspace(
                source, root / "cache", plan, "marker-test"
            )
            self.assertEqual(rebuilt, workspace)
            self.assertFalse(stale.exists())
            self.assertTrue(marker.is_file())
            self.assertTrue(
                deps.state_matches(marker, deps._build_lock_payload(plan))
            )


class SubprocessEnvironmentTests(unittest.TestCase):
    def test_sanitizer_removes_secrets_python_and_pip_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary).resolve() / "pip-cache"
            cache.mkdir()
            source = {
                "PATH": "/usr/bin:/bin",
                "HOME": "/home/modly",
                "LANG": "C.UTF-8",
                "CUDA_HOME": "/usr/local/cuda-12.4",
                "CXX": "/usr/bin/g++",
                "PIP_CACHE_DIR": str(cache),
                "PIP_INDEX_URL": "https://evil.invalid/simple",
                "PIP_EXTRA_INDEX_URL": "https://evil.invalid/extra",
                "PIP_CONFIG_FILE": "/tmp/attacker-pip.conf",
                "PIP_TRUSTED_HOST": "evil.invalid",
                "PYTHONPATH": "/tmp/injected",
                "PYTHONHOME": "/tmp/python-home",
                "HF_TOKEN": "secret",
                "DATABASE_PASSWORD": "secret",
                "SSH_AUTH_SOCK": "/tmp/agent.sock",
                "AWS_PROFILE": "private",
                "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/google.json",
                "HTTP_PROXY": "http://user:password@proxy.invalid:8080",
                "HTTPS_PROXY": "http://proxy.invalid:8080",
                "NO_PROXY": "localhost,127.0.0.1",
                "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
            }
            network = deps.sanitize_subprocess_environment(
                source, allow_network=True
            )
            local = deps.sanitize_subprocess_environment(
                source, allow_network=False
            )
            pip_environment = deps.sanitize_subprocess_environment(
                source, allow_network=True, for_pip=True
            )

        self.assertEqual(network["PIP_CACHE_DIR"], str(cache))
        self.assertEqual(network["HTTPS_PROXY"], "http://proxy.invalid:8080")
        self.assertEqual(network["NO_PROXY"], "localhost,127.0.0.1")
        self.assertNotIn("HTTP_PROXY", network)
        for forbidden in (
            "PIP_INDEX_URL",
            "PIP_EXTRA_INDEX_URL",
            "PIP_CONFIG_FILE",
            "PIP_TRUSTED_HOST",
            "PYTHONPATH",
            "PYTHONHOME",
            "HF_TOKEN",
            "DATABASE_PASSWORD",
            "SSH_AUTH_SOCK",
            "AWS_PROFILE",
            "GOOGLE_APPLICATION_CREDENTIALS",
        ):
            self.assertNotIn(forbidden, network)
        self.assertNotIn("HTTPS_PROXY", local)
        self.assertNotIn("NO_PROXY", local)
        self.assertEqual(pip_environment["PIP_CONFIG_FILE"], deps.os.devnull)

    def test_isolated_pip_keeps_only_valid_shared_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            cache = root / "pip-cache"
            cache.mkdir()
            env = deps.sanitize_subprocess_environment(
                {
                    "PATH": "/usr/bin:/bin",
                    "PIP_CACHE_DIR": str(cache),
                    "PIP_INDEX_URL": "https://evil.invalid/simple",
                },
                allow_network=True,
            )
            command = deps.isolated_pip_command(
                Path("/venv/python"),
                ["install", "--index-url", deps.PYPI_INDEX, "wheel==0.45.1"],
                env,
            )
            alias = root / "alias-cache"
            try:
                alias.symlink_to(cache, target_is_directory=True)
            except OSError:
                alias = None
            alias_env = {"PIP_CACHE_DIR": str(alias)} if alias is not None else {}
            alias_command = deps.isolated_pip_command(
                Path("/venv/python"), ["check"], alias_env
            )

        self.assertEqual(command[1:3], ["-m", "pip"])
        self.assertIn("--isolated", command)
        self.assertIn("--disable-pip-version-check", command)
        self.assertIn("--no-input", command)
        cache_position = command.index("--cache-dir")
        self.assertEqual(command[cache_position + 1], str(cache))
        self.assertNotIn("https://evil.invalid/simple", command)
        self.assertNotIn("--cache-dir", alias_command)

    def test_install_falls_back_to_owned_pip_cache_without_following_alias(self) -> None:
        plan = deps.select_dependency_plan(
            context(arch="arm64", sm=90, cuda=126), "portable"
        )
        commands: list[list[str]] = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            cache = root / "cache"
            outside = root / "outside"
            cache.mkdir()
            outside.mkdir()
            marker = outside / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            (cache / "pip").symlink_to(outside, target_is_directory=True)
            inherited = {
                "PATH": "/usr/bin:/bin",
                "PIP_CACHE_DIR": str(cache / "pip"),
            }
            with mock.patch.dict(deps.os.environ, inherited, clear=True), mock.patch.object(
                deps, "_assert_target_python"
            ), mock.patch.object(deps, "_validate_linux_glibc"), mock.patch.object(
                deps, "verify_dependencies", return_value={"ok": True}
            ):
                constraint = deps.install_dependencies(
                    Path(__file__),
                    plan,
                    cache,
                    runner=lambda command, _env: commands.append(list(command)),
                    log=lambda _message: None,
                )
            self.assertTrue(marker.is_file())
            self.assertFalse((cache / "pip").is_symlink())
            self.assertTrue(constraint.is_file())
            for command in commands:
                position = command.index("--cache-dir")
                self.assertEqual(command[position + 1], str(cache / "pip"))


class PatchTests(unittest.TestCase):
    def test_flexgemm_patch_removes_home_cache_from_setup_and_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "flex_gemm").mkdir()
            (root / "setup.py").write_text(
                'os.makedirs(os.path.expanduser("~/.flex_gemm"), exist_ok=True)\n'
                'src_cache_path = os.path.join(ROOT, "autotune_cache.json")\n'
                'dst_cache_path = os.path.expanduser("~/.flex_gemm/autotune_cache.json")\n',
                encoding="utf-8",
            )
            (root / "flex_gemm" / "__init__.py").write_text(
                "AUTOTUNE_CACHE_PATH = os.environ.get(\n"
                "    'FLEX_GEMM_AUTOTUNE_CACHE_PATH',\n"
                "    os.path.expanduser('~/.flex_gemm/autotune_cache.json')\n"
                ")\n",
                encoding="utf-8",
            )
            deps._patch_flexgemm(root)
            combined = (root / "setup.py").read_text() + (
                root / "flex_gemm" / "__init__.py"
            ).read_text()
            self.assertNotIn("expanduser", combined)
            self.assertIn("MODLY_LATO2_CACHE_DIR", combined)

    def test_cuda_124_toolkit_is_selected_when_newer_nvcc_is_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            newer = root / "cuda-13.0"
            pinned = root / "cuda-12.4"
            for toolkit in (newer, pinned):
                (toolkit / "bin").mkdir(parents=True)
                (toolkit / "bin" / "nvcc").touch()

            def fake_run(command, **_kwargs):
                version = "13.0" if str(newer) in command[0] else "12.4"
                return subprocess.CompletedProcess(
                    command, 0, stdout=f"Cuda compilation tools, release {version}, V{version}.0\n", stderr=""
                )

            with mock.patch.object(
                deps.shutil, "which", return_value=str(newer / "bin" / "nvcc")
            ), mock.patch.object(deps.subprocess, "run", side_effect=fake_run):
                selected = deps._find_cuda_home(
                    {
                        "PATH": str(newer / "bin"),
                        "CUDA_HOME": str(newer),
                        "CUDA_PATH": str(pinned),
                    }
                )
            self.assertEqual(selected, pinned.resolve())

    def test_native_build_env_retains_only_credential_free_network_settings(self) -> None:
        plan = deps.select_dependency_plan(context(sm=86), "exact-upstream")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            cuda = root / "cuda-12.4"
            (cuda / "bin").mkdir(parents=True)
            nvcc = cuda / "bin" / "nvcc"
            nvcc.touch()
            completed = subprocess.CompletedProcess(
                [str(nvcc), "--version"],
                0,
                stdout="Cuda compilation tools, release 12.4, V12.4.0\n",
                stderr="",
            )
            with mock.patch.object(
                deps, "_find_cuda_home", return_value=cuda
            ), mock.patch.object(
                deps,
                "_linux_cxx_environment",
                return_value={"PATH": "/usr/bin:/bin", "CXX": "/usr/bin/g++"},
            ), mock.patch.object(deps.subprocess, "run", return_value=completed):
                env = deps.native_build_environment(
                    plan,
                    root / "cache",
                    base_env={
                        "PATH": "/usr/bin:/bin",
                        "HTTPS_PROXY": "http://proxy.invalid:8080",
                        "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
                        "HTTP_PROXY": "http://user:password@proxy.invalid:8080",
                        "API_TOKEN": "secret",
                    },
                )
        self.assertEqual(env["HTTPS_PROXY"], "http://proxy.invalid:8080")
        self.assertEqual(
            env["SSL_CERT_FILE"], "/etc/ssl/certs/ca-certificates.crt"
        )
        self.assertNotIn("HTTP_PROXY", env)
        self.assertNotIn("API_TOKEN", env)


class SourceCacheIntegrityTests(unittest.TestCase):
    def test_native_marker_cannot_self_attest_modified_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "native-sources" / "fixture"
            root.mkdir(parents=True)
            (root / "setup.py").write_text("value = 1\n", encoding="utf-8")
            deps.write_state(root / "source-lock.json", deps._source_marker_payload("linux"))
            trusted_digest = deps.inventory_tree(root).digest
            with mock.patch.object(
                deps, "_source_archives_are_trusted", return_value=True
            ), mock.patch.dict(
                deps.NATIVE_SOURCE_TREE_SHA256,
                {"linux": trusted_digest},
                clear=True,
            ):
                self.assertTrue(deps._native_sources_valid(root, "linux"))
                (root / "setup.py").write_text("value = 2\n", encoding="utf-8")
                # Rewriting the mutable marker to its expected JSON does not
                # alter the extension-controlled complete-tree digest.
                deps.write_state(
                    root / "source-lock.json", deps._source_marker_payload("linux")
                )
                self.assertFalse(deps._native_sources_valid(root, "linux"))

    def test_portable_cpu_marker_cannot_self_attest_modified_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "portable-cpu-sources" / "fixture"
            root.mkdir(parents=True)
            (root / "operator.cpp").write_text("int value = 1;\n", encoding="utf-8")
            deps.write_state(
                root / "source-lock.json", deps._portable_cpu_marker_payload()
            )
            trusted_digest = deps.inventory_tree(root).digest
            with mock.patch.object(
                deps, "_source_archives_are_trusted", return_value=True
            ), mock.patch.object(
                deps, "PORTABLE_CPU_SOURCE_TREE_SHA256", trusted_digest
            ):
                self.assertTrue(deps._portable_cpu_sources_valid(root))
                (root / "operator.cpp").write_text(
                    "int value = 9;\n", encoding="utf-8"
                )
                deps.write_state(
                    root / "source-lock.json", deps._portable_cpu_marker_payload()
                )
                self.assertFalse(deps._portable_cpu_sources_valid(root))

    def test_source_cache_alias_is_removed_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            cache = base / "cache"
            cache.mkdir()
            external = base / "external"
            external.mkdir()
            sentinel = external / "keep.txt"
            sentinel.write_text("keep\n", encoding="utf-8")
            alias = cache / "native-sources"
            try:
                alias.symlink_to(external, target_is_directory=True)
            except OSError:
                self.skipTest("directory symlinks unavailable")
            repaired = deps._prepare_source_cache_directory(cache, "native-sources")
            self.assertTrue(repaired.is_dir())
            self.assertFalse(repaired.is_symlink())
            self.assertTrue(sentinel.is_file())

    def test_pinned_source_tree_digests_are_real_sha256(self) -> None:
        for digest in (
            *deps.NATIVE_SOURCE_TREE_SHA256.values(),
            deps.PORTABLE_CPU_SOURCE_TREE_SHA256,
        ):
            self.assertRegex(digest, r"^[0-9a-f]{64}$")


class InstallPlanTests(unittest.TestCase):
    def test_exact_install_has_every_native_stage_and_no_git_command(self) -> None:
        plan = deps.select_dependency_plan(context(sm=86), "exact-upstream")
        commands: list[list[str]] = []
        environments: list[dict[str, str]] = []

        def runner(command, env):
            commands.append(list(command))
            environments.append(dict(env or {}))

        fake_sources = deps.NativeSources(
            Path("/cache/native"),
            Path("/cache/native/nvdiffrast"),
            Path("/cache/native/CuMesh"),
            Path("/cache/native/FlexGEMM"),
            Path("/cache/native/o-voxel"),
        )
        inherited = {
            "PATH": "/usr/bin:/bin",
            "HTTPS_PROXY": "http://proxy.invalid:8080",
            "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
            "PIP_INDEX_URL": "https://evil.invalid/simple",
            "PIP_CONFIG_FILE": "/tmp/evil-pip.conf",
            "PYTHONPATH": "/tmp/injected",
            "HF_TOKEN": "secret",
        }
        with mock.patch.dict(deps.os.environ, inherited, clear=True), mock.patch.object(
            deps, "_assert_target_python"
        ), mock.patch.object(deps, "_validate_linux_glibc"), mock.patch.object(
            deps,
            "native_build_environment",
            return_value={
                "PATH": "/usr/bin:/bin",
                "HTTPS_PROXY": "http://proxy.invalid:8080",
                "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
            },
        ), mock.patch.object(
            deps, "prepare_native_sources", return_value=fake_sources
        ), mock.patch.object(
            deps,
            "prepare_build_workspace",
            side_effect=lambda source, _cache, _plan, _purpose: source,
        ), mock.patch.object(deps, "verify_dependencies", return_value={"ok": True}):
            deps.install_dependencies(
                Path(__file__), plan, Path("/cache"), runner=runner, log=lambda _: None
            )
        flat = "\n".join(" ".join(command) for command in commands)
        for required in (
            "spconv-cu124==2.3.8",
            "torch-scatter==2.1.2+pt26cu124",
            "xformers==0.0.29.post2",
            "flash-attn==2.7.4.post1",
            "filelock==3.20.0",
            "/cache/native/nvdiffrast",
            "/cache/native/CuMesh",
            "/cache/native/FlexGEMM",
            "/cache/native/o-voxel",
        ):
            self.assertIn(required, flat)
        self.assertNotIn("git+", flat)
        self.assertTrue(commands[-1][-1] == "check")
        self.assertTrue(all("--isolated" in command for command in commands))
        self.assertTrue(all("--no-input" in command for command in commands))
        for command, environment in zip(commands, environments):
            local_source_build = any(
                source in command
                for source in (
                    "/cache/native/nvdiffrast",
                    "/cache/native/CuMesh",
                    "/cache/native/FlexGEMM",
                    "/cache/native/o-voxel",
                )
            )
            if "install" in command:
                self.assertTrue(
                    "--index-url" in command or "--no-index" in command,
                    command,
                )
                self.assertIn("--constraint", command)
                self.assertTrue(
                    any(part.endswith(".txt") for part in command), command
                )
            for forbidden in (
                "PIP_INDEX_URL",
                "PYTHONPATH",
                "HF_TOKEN",
            ):
                self.assertNotIn(forbidden, environment)
            self.assertEqual(environment.get("PIP_CONFIG_FILE"), deps.os.devnull)
            if local_source_build:
                self.assertIn("--no-index", command)
            if (
                "install" in command
                and not local_source_build
                and deps.FLASH_ATTN_REQUIREMENT not in command
            ):
                self.assertIn("--only-binary=:all:", command)
            if local_source_build or command[-1] == "check":
                self.assertNotIn("HTTPS_PROXY", environment)
                self.assertNotIn("SSL_CERT_FILE", environment)
            else:
                self.assertEqual(
                    environment.get("HTTPS_PROXY"),
                    "http://proxy.invalid:8080",
                )
                self.assertEqual(
                    environment.get("SSL_CERT_FILE"),
                    "/etc/ssl/certs/ca-certificates.crt",
                )
        self.assertIn(deps.PYPI_INDEX, commands[0])
        self.assertIn(deps.PYPI_INDEX, commands[1])

    def test_cpu_build_environment_accepts_exact_and_limits_parallelism(self) -> None:
        plan = deps.select_dependency_plan(context(sm=86), "exact-upstream")
        with tempfile.TemporaryDirectory() as temporary:
            env = deps.cpu_build_environment(
                plan,
                Path(temporary),
                base_env={"PATH": "/usr/bin:/bin"},
            )
        self.assertEqual(env["MAX_JOBS"], "1")
        self.assertIn("torch-extensions", env["TORCH_EXTENSIONS_DIR"])
        self.assertEqual(
            Path(env["TORCH_EXTENSIONS_DIR"]).name,
            deps.dependency_lock_digest(plan)[: deps._BUILD_LOCK_PREFIX_LENGTH],
        )

    def test_filelock_is_pinned_and_checked_as_a_direct_runtime_import(self) -> None:
        plan = deps.select_dependency_plan(context(sm=86), "exact-upstream")
        self.assertIn("filelock==3.20.0", deps.BASE_REQUIREMENTS)
        self.assertEqual(
            deps.expected_distribution_versions(plan)["filelock"], "3.20.0"
        )

    def test_cpu_extension_smoke_executes_real_tetrahedron_operator(self) -> None:
        plan = deps.select_dependency_plan(
            context(system="win32", arch="x64", sm=75), "portable"
        )
        completed = subprocess_result = mock.Mock()
        subprocess_result.stdout = json.dumps(
            {
                "ok": True,
                "version": deps.OVOXEL_CPU_VERSION,
                "buildIdentity": deps.OVOXEL_CPU_BUILD_IDENTITY,
                "templateTreeSha256": deps.TEMPLATE_TREE_SHA256,
                "licensesVerified": sorted(deps.LICENSE_SOURCE_SPECS),
                "voxelCount": 12,
            }
        )
        calls: list[list[str]] = []
        environments: list[dict[str, str]] = []

        def fake_run(command, **kwargs):
            calls.append(list(command))
            environments.append(dict(kwargs["env"]))
            if command[-1] == "check":
                result = mock.Mock()
                result.stdout = ""
                return result
            return completed

        with mock.patch.object(
            deps, "cpu_build_environment", return_value={"PATH": ""}
        ), mock.patch.object(deps.subprocess, "run", side_effect=fake_run):
            result = deps.verify_portable_cpu_extension(
                Path("/venv/python"), plan, Path("/cache")
            )
        self.assertEqual(result["voxelCount"], 12)
        smoke_script = calls[1][-1]
        self.assertIn("mesh_to_flexible_dual_grid_cpu", smoke_script)
        self.assertIn("torch.isfinite", smoke_script)
        self.assertIn('metadata.get_all("License-File")', smoke_script)
        self.assertIn("hashlib.sha256", smoke_script)
        smoke_identity = json.loads(
            environments[1]["MODLY_LATO2_OVOXEL_CPU_IDENTITY"]
        )
        self.assertEqual(smoke_identity["version"], deps.OVOXEL_CPU_VERSION)
        self.assertEqual(
            smoke_identity["buildIdentity"], deps.OVOXEL_CPU_BUILD_IDENTITY
        )
        self.assertEqual(
            smoke_identity["licenseSha256"],
            {
                name: digest
                for name, (_relative, digest) in deps.LICENSE_SOURCE_SPECS.items()
            },
        )

    def test_exact_smoke_executes_cuda_dependency_kernels(self) -> None:
        plan = deps.select_dependency_plan(context(sm=86), "exact-upstream")
        completed = mock.Mock()
        completed.stdout = json.dumps(
            {
                "ok": True,
                "capability": [8, 6],
                "native": True,
                "attention": "flash_attn",
            }
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            deps.subprocess, "run", return_value=completed
        ) as run:
            deps.verify_dependencies(Path("/venv/python"), plan, Path(temporary))
        smoke_script = run.call_args.args[0][-1]
        for token in (
            "import filelock",
            "spconv.SubMConv3d",
            "torch_scatter.scatter_mean",
            "ovoxel_cuda.mesh_to_flexible_dual_grid_cpu",
            "xops.memory_efficient_attention",
            "flash_attn.flash_attn_func",
            "torch.cuda.synchronize",
        ):
            self.assertIn(token, smoke_script)


if __name__ == "__main__":
    unittest.main()
