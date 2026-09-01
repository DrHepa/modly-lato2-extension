from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import stat
import sys
import tempfile
import unittest
from unittest import mock

from lato2_modly import dependencies as deps


def python_fingerprint(minor: int, *, arch: str = "x86_64") -> dict[str, object]:
    return {
        "implementation": "cpython",
        "version": [3, minor],
        "cache_tag": f"cpython-3{minor}",
        "abiflags": "",
        "soabi": f"cpython-3{minor}-{arch}-linux-gnu",
        "platform": f"linux-{arch}",
        "machine": arch,
        "pointer_bits": 64,
    }


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


def fake_venv_python(root: Path, *, symlink: bool = False) -> Path:
    venv = root / "venv.__modly_staging"
    (venv / "pyvenv.cfg").parent.mkdir(parents=True, exist_ok=True)
    (venv / "pyvenv.cfg").write_text(
        "include-system-site-packages = false\n", encoding="utf-8"
    )
    python = venv / "bin/python"
    python.parent.mkdir()
    if symlink:
        try:
            python.symlink_to(Path(sys.executable).resolve())
        except OSError as exc:
            raise unittest.SkipTest(f"executable symlinks unavailable: {exc}") from exc
    else:
        python.touch()
    return python


def _record_digest(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode(
        "ascii"
    )


def compact_cusparselt_contract(
    *,
    metadata_name: str = "nvidia-cusparselt-cu12",
    metadata_version: str = "0.7.1",
    elf_machine: int = 183,
    empty_library: bool = False,
) -> tuple[object, dict[str, bytes], bytes, bytes]:
    dist_info = "nvidia_cusparselt_cu12-0.7.1.dist-info"
    package_root = "nvidia/cusparselt"
    library = b"" if empty_library else bytearray(64)
    if isinstance(library, bytearray):
        library[:6] = b"\x7fELF\x02\x01"
        library[6] = 1
        library[18:20] = elf_machine.to_bytes(2, "little")
        library = bytes(library)
    original_wheel = (
        b"Wheel-Version: 1.0\n"
        b"Generator: setuptools (75.8.0)\n"
        b"Root-Is-Purelib: true\n"
        b"Tag: py3-none-manylinux2014_sbsa\n\n"
    )
    normalized_wheel = original_wheel.replace(
        b"manylinux2014_sbsa", b"manylinux2014_aarch64"
    )
    contents = {
        f"{package_root}/LICENSE.txt": b"license\n",
        f"{package_root}/include/cusparseLt.h": b"header\n",
        f"{package_root}/lib/libcusparseLt.so.0": library,
        f"{dist_info}/INSTALLER": b"pip\n",
        f"{dist_info}/METADATA": (
            b"Metadata-Version: 2.1\n"
            + f"Name: {metadata_name}\n".encode("ascii")
            + f"Version: {metadata_version}\n".encode("ascii")
        ),
        f"{dist_info}/top_level.txt": b"nvidia/cusparselt\n",
    }
    wheel_path = f"{dist_info}/WHEEL"
    record_path = f"{dist_info}/RECORD"
    record_order = (
        f"{package_root}/LICENSE.txt",
        f"{package_root}/include/cusparseLt.h",
        f"{package_root}/lib/libcusparseLt.so.0",
        f"{dist_info}/INSTALLER",
        f"{dist_info}/METADATA",
        record_path,
        wheel_path,
        f"{dist_info}/top_level.txt",
    )

    def make_record(wheel: bytes) -> bytes:
        rows = []
        for relative in record_order:
            if relative == record_path:
                rows.append(f"{relative},,")
                continue
            data = wheel if relative == wheel_path else contents[relative]
            rows.append(f"{relative},sha256={_record_digest(data)},{len(data)}")
        return ("\r\n".join(rows) + "\r\n").encode("ascii")

    original_record = make_record(original_wheel)
    normalized_record = make_record(normalized_wheel)
    contract = deps._CusparseLtNormalizationContract(
        schema="test.cusparselt-normalization.v1",
        distribution="nvidia-cusparselt-cu12",
        version="0.7.1",
        dist_info=dist_info,
        package_root=package_root,
        library_relative=f"{package_root}/lib/libcusparseLt.so.0",
        metadata_relative=f"{dist_info}/METADATA",
        wheel_relative=wheel_path,
        record_relative=record_path,
        package_directories=("include", "lib"),
        record_order=record_order,
        fixed_files=tuple(
            (relative, hashlib.sha256(data).hexdigest(), len(data))
            for relative, data in contents.items()
        ),
        original_wheel=original_wheel,
        normalized_wheel=normalized_wheel,
        original_record_sha256=hashlib.sha256(original_record).hexdigest(),
        normalized_record_sha256=hashlib.sha256(normalized_record).hexdigest(),
    )
    return contract, contents, original_record, normalized_record


def fake_cusparselt_distribution(
    root: Path,
    minor: int,
    *,
    bundle: tuple[object, dict[str, bytes], bytes, bytes] | None = None,
    state: str = "original",
) -> tuple[Path, object, Path, bytes, bytes]:
    contract, contents, original_record, normalized_record = (
        compact_cusparselt_contract() if bundle is None else bundle
    )
    python = fake_venv_python(root)
    site_packages = (
        python.parent.parent / "lib" / f"python3.{minor}" / "site-packages"
    )
    for relative, data in contents.items():
        target = site_packages / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    wheel = site_packages / contract.wheel_relative
    record = site_packages / contract.record_relative
    wheel.parent.mkdir(parents=True, exist_ok=True)
    if state in {"normalized", "hybrid-wheel"}:
        wheel.write_bytes(contract.normalized_wheel)
    else:
        wheel.write_bytes(contract.original_wheel)
    if state in {"normalized", "hybrid-record"}:
        record.write_bytes(normalized_record)
    else:
        record.write_bytes(original_record)
    return python, contract, site_packages, original_record, normalized_record


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


class CusparseLtNormalizationTests(unittest.TestCase):
    @staticmethod
    def plan(minor: int = 12):
        return deps.select_dependency_plan(
            context(arch="aarch64", sm=120, cuda=128),
            "portable",
            interpreter_fingerprint=python_fingerprint(minor, arch="aarch64"),
        )

    def test_lock_and_state_record_the_exact_pre_and_post_identity(self) -> None:
        for minor in (11, 12):
            with self.subTest(minor=minor):
                plan = self.plan(minor)
                identity = deps.dependency_lock_payload(plan)[
                    "installedMetadataNormalization"
                ]
                self.assertEqual(identity["distribution"], "nvidia-cusparselt-cu12")
                self.assertEqual(identity["version"], "0.7.1")
                self.assertEqual(identity["pythonLanes"], ["cp311", "cp312"])
                self.assertEqual(
                    identity["wheel"]["beforeSha256"],
                    "6277516b4579f35a685c012370b1634b4d7f3a904f2aa91885c04006ab56253c",
                )
                self.assertEqual(
                    identity["wheel"]["afterSha256"],
                    "080008623ebbde6f36b884ca554ca80d6dcd11fd54f653bda9879669674afb0f",
                )
                state = deps.dependency_state_payload(
                    plan, python_fingerprint(minor, arch="aarch64")
                )
                self.assertEqual(state["installedMetadataNormalization"], identity)

    def test_production_contract_snapshots_every_fixed_file_and_record_hash(self) -> None:
        contract = deps._CUSPARSELT_NORMALIZATION_CONTRACT
        self.assertEqual(
            contract.fixed_files,
            (
                (
                    "nvidia/cusparselt/LICENSE.txt",
                    "e8d158885a681b95ec7a6fc06dd8d4a52989f374cb1380c8a4c8fb27fd3d5d5e",
                    17948,
                ),
                (
                    "nvidia/cusparselt/include/cusparseLt.h",
                    "74580d3104ed58e1708d2ee746f65c2a9e1557f91b1d158d2d93c1591e118c38",
                    17876,
                ),
                (
                    "nvidia/cusparselt/lib/libcusparseLt.so.0",
                    "2c677e678d1955a6dedd66274dfe4cc0f930fea6421f1b8a5ec08cbb1ea18b17",
                    440496193,
                ),
                (
                    "nvidia_cusparselt_cu12-0.7.1.dist-info/INSTALLER",
                    "ceebae7b8927a3227e5303cf5e0f1f7b34bb542ad7250ac03fbcde36ec2f1508",
                    4,
                ),
                (
                    "nvidia_cusparselt_cu12-0.7.1.dist-info/METADATA",
                    "b264cea6951b70b52a3fd2ffaa88f9109eaf379633aa8d6e0832ecc92ddf6fba",
                    6974,
                ),
                (
                    "nvidia_cusparselt_cu12-0.7.1.dist-info/top_level.txt",
                    "a1f202f9d6cad2cdb68cde79b207d5a5b847593eb765d24b59e49be8aff5f812",
                    18,
                ),
            ),
        )
        original_record, normalized_record = deps._validate_cusparselt_contract(
            contract
        )
        self.assertEqual(len(original_record), 754)
        self.assertEqual(len(normalized_record), 754)
        self.assertEqual(
            hashlib.sha256(original_record).hexdigest(),
            "e02265be65d5f1aab74b97ffd2d53ca2ccf9d41ba0ca9cddf8d9b4bd34262425",
        )
        self.assertEqual(
            hashlib.sha256(normalized_record).hexdigest(),
            "54b7063009a7cb86f1a0b2dd0bc0af777f8946440084365671e1745a1d61a976",
        )

    def test_normalizes_only_wheel_and_its_record_row_for_both_python_lanes(self) -> None:
        for minor in (11, 12):
            with self.subTest(minor=minor), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                python, contract, site, original_record, normalized_record = (
                    fake_cusparselt_distribution(root, minor)
                )
                original_metadata = (site / contract.metadata_relative).read_bytes()
                with mock.patch.object(
                    deps, "_CUSPARSELT_NORMALIZATION_CONTRACT", contract
                ):
                    report = deps.normalize_cusparselt_metadata(
                        python,
                        self.plan(minor),
                        _cdll_loader=lambda _path: object(),
                    )
                self.assertTrue(report["applied"])
                self.assertEqual(
                    (site / contract.wheel_relative).read_bytes(),
                    contract.normalized_wheel,
                )
                self.assertEqual(
                    (site / contract.record_relative).read_bytes(), normalized_record
                )
                self.assertNotEqual(original_record, normalized_record)
                self.assertEqual(
                    (site / contract.metadata_relative).read_bytes(), original_metadata
                )
                changed_rows = [
                    (before, after)
                    for before, after in zip(
                        original_record.splitlines(), normalized_record.splitlines()
                    )
                    if before != after
                ]
                self.assertEqual(len(changed_rows), 1)
                self.assertIn(b".dist-info/WHEEL,", changed_rows[0][0])

    def test_exact_final_state_is_idempotent_without_replacing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python, contract, _site, _before, _after = fake_cusparselt_distribution(
                root, 12, state="normalized"
            )
            with mock.patch.object(
                deps, "_CUSPARSELT_NORMALIZATION_CONTRACT", contract
            ), mock.patch.object(deps.os, "replace", wraps=os.replace) as replace_file:
                report = deps.normalize_cusparselt_metadata(
                    python,
                    self.plan(),
                    _cdll_loader=lambda _path: object(),
                )
            self.assertFalse(report["applied"])
            replace_file.assert_not_called()

    def test_library_swap_cannot_change_the_inode_loaded_after_hashing(self) -> None:
        loaded: list[bytes] = []
        loader_paths: list[str] = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python, contract, site, _before, _after = fake_cusparselt_distribution(
                root, 12
            )
            library = site / contract.library_relative
            original = library.read_bytes()
            replacement = root / "same-size-replacement.so"
            replacement.write_bytes(b"X" * len(original))

            def racing_loader(path):
                loader_paths.append(str(path))
                os.replace(replacement, library)
                loaded.append(Path(path).read_bytes())
                return object()

            with mock.patch.object(
                deps, "_CUSPARSELT_NORMALIZATION_CONTRACT", contract
            ), self.assertRaises(deps.DependencyError):
                deps.normalize_cusparselt_metadata(
                    python, self.plan(), _cdll_loader=racing_loader
                )
        self.assertEqual(loaded, [original])
        self.assertEqual(len(loader_paths), 1)
        self.assertTrue(loader_paths[0].startswith("/proc/self/fd/"))

    def test_dist_info_swap_cannot_redirect_atomic_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python, contract, site, _before, _after = fake_cusparselt_distribution(
                root, 12
            )
            dist_info = site / contract.dist_info
            parked = root / "parked-dist-info"
            outside = root / "outside-dist-info"
            outside.mkdir()
            outside_wheel = outside / "WHEEL"
            outside_wheel.write_bytes(b"outside-sentinel")
            original_replace = os.replace
            raced = False
            replace_kwargs: list[dict[str, object]] = []

            def racing_replace(source, destination, *args, **kwargs):
                nonlocal raced
                replace_kwargs.append(dict(kwargs))
                if not raced:
                    raced = True
                    dist_info.rename(parked)
                    try:
                        dist_info.symlink_to(outside, target_is_directory=True)
                    except OSError as exc:
                        raise unittest.SkipTest(
                            f"directory symlinks unavailable: {exc}"
                        ) from exc
                    source_name = Path(source).name
                    (outside / source_name).write_bytes(
                        (parked / source_name).read_bytes()
                    )
                return original_replace(source, destination, *args, **kwargs)

            with mock.patch.object(
                deps, "_CUSPARSELT_NORMALIZATION_CONTRACT", contract
            ), mock.patch.object(deps.os, "replace", side_effect=racing_replace), self.assertRaises(
                deps.DependencyError
            ):
                deps.normalize_cusparselt_metadata(
                    python, self.plan(), _cdll_loader=lambda _path: object()
                )
            self.assertTrue(raced)
            self.assertEqual(outside_wheel.read_bytes(), b"outside-sentinel")
            self.assertTrue(replace_kwargs)
            self.assertEqual(
                replace_kwargs[0]["src_dir_fd"], replace_kwargs[0]["dst_dir_fd"]
            )

    def test_synthetic_pip_check_returns_zero_only_after_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python, contract, site, _before, _after = fake_cusparselt_distribution(
                root, 12
            )
            wheel = site / contract.wheel_relative
            python.write_text(
                f"#!{sys.executable}\n"
                "from pathlib import Path\n"
                "import sys\n"
                f"wheel = Path({str(wheel)!r})\n"
                f"expected = {contract.normalized_wheel!r}\n"
                "sys.exit(0 if sys.argv[-1] == 'check' and wheel.read_bytes() == expected else 9)\n",
                encoding="utf-8",
            )
            python.chmod(0o755)
            command = deps.isolated_pip_command(python, ["check"], {})
            self.assertNotEqual(
                subprocess.run(command, check=False).returncode,
                0,
            )
            with mock.patch.object(
                deps, "_CUSPARSELT_NORMALIZATION_CONTRACT", contract
            ):
                deps.normalize_cusparselt_metadata(
                    python, self.plan(), _cdll_loader=lambda _path: object()
                )
            subprocess.run(command, check=True)

    def test_non_applicable_platform_architecture_and_torch_lane_are_noops(self) -> None:
        plans = (
            replace(self.plan(), system="win32"),
            deps.select_dependency_plan(
                context(arch="x64", sm=120, cuda=128),
                "portable",
                interpreter_fingerprint=python_fingerprint(12),
            ),
            deps.select_dependency_plan(
                context(arch="aarch64", sm=90, cuda=126),
                "portable",
                interpreter_fingerprint=python_fingerprint(12, arch="aarch64"),
            ),
        )
        for plan in plans:
            with self.subTest(system=plan.system, arch=plan.arch, lane=plan.torch_lane):
                self.assertIsNone(
                    deps.normalize_cusparselt_metadata(Path("/missing/bin/python"), plan)
                )

    def _assert_rejected(
        self,
        *,
        bundle=None,
        state: str = "original",
        mutate=None,
        loader=None,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python, contract, site, _before, _after = fake_cusparselt_distribution(
                root, 12, bundle=bundle, state=state
            )
            if mutate is not None:
                mutate(root, site, contract)
            with mock.patch.object(
                deps, "_CUSPARSELT_NORMALIZATION_CONTRACT", contract
            ), self.assertRaises(deps.DependencyError):
                deps.normalize_cusparselt_metadata(
                    python,
                    self.plan(),
                    _cdll_loader=(lambda _path: object()) if loader is None else loader,
                )

    def test_rejects_wrong_distribution_version_tag_hash_and_file_set(self) -> None:
        self._assert_rejected(
            bundle=compact_cusparselt_contract(metadata_name="different-package")
        )
        self._assert_rejected(
            bundle=compact_cusparselt_contract(metadata_version="0.7.0")
        )
        self._assert_rejected(
            mutate=lambda _root, site, contract: (
                site / contract.wheel_relative
            ).write_bytes(contract.original_wheel.replace(b"sbsa", b"arm64"))
        )
        self._assert_rejected(
            mutate=lambda _root, site, contract: (
                site / contract.package_root / "LICENSE.txt"
            ).write_bytes(b"tampered\n")
        )

        def add_file(_root, site, contract):
            (site / contract.package_root / "unexpected.txt").write_text(
                "unexpected", encoding="utf-8"
            )

        self._assert_rejected(mutate=add_file)

    def test_rejects_symlinks_hardlinks_record_escape_and_duplicates(self) -> None:
        def install_symlink(root, site, contract):
            target = site / contract.package_root / "include/cusparseLt.h"
            target.unlink()
            outside = root / "outside-header"
            outside.write_bytes(b"header\n")
            try:
                target.symlink_to(outside)
            except OSError as exc:
                raise unittest.SkipTest(f"symlinks unavailable: {exc}") from exc

        self._assert_rejected(mutate=install_symlink)

        def hardlink_file(root, site, contract):
            target = site / contract.package_root / "LICENSE.txt"
            try:
                os.link(target, root / "outside-hardlink")
            except OSError as exc:
                raise unittest.SkipTest(f"hardlinks unavailable: {exc}") from exc

        self._assert_rejected(mutate=hardlink_file)

        def unsafe_record(_root, site, contract):
            record = site / contract.record_relative
            record.write_bytes(b"../escape,sha256=AAAA,1\r\n" + record.read_bytes())

        self._assert_rejected(mutate=unsafe_record)

        def duplicate_record(_root, site, contract):
            record = site / contract.record_relative
            first = record.read_bytes().splitlines(keepends=True)[0]
            record.write_bytes(first + record.read_bytes())

        self._assert_rejected(mutate=duplicate_record)

    def test_rejects_empty_or_wrong_elf_and_dlopen_failure(self) -> None:
        self._assert_rejected(bundle=compact_cusparselt_contract(empty_library=True))
        self._assert_rejected(bundle=compact_cusparselt_contract(elf_machine=62))

        def fail_dlopen(_path):
            raise OSError("cannot load")

        self._assert_rejected(loader=fail_dlopen)

    def test_rejects_both_hybrid_states(self) -> None:
        for state in ("hybrid-wheel", "hybrid-record"):
            with self.subTest(state=state):
                self._assert_rejected(state=state)

    def test_install_normalizes_after_torch_and_before_mandatory_pip_check(self) -> None:
        events: list[str] = []
        plan = self.plan()

        def runner(command, _env):
            if plan.torch_requirements[0] in command:
                events.append("torch")
            if command[-1] == "check":
                events.append("pip-check")
                raise subprocess.CalledProcessError(1, command)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python = fake_venv_python(root)
            with mock.patch.object(deps, "_assert_target_python"), mock.patch.object(
                deps, "_validate_linux_glibc"
            ), mock.patch.object(
                deps,
                "normalize_cusparselt_metadata",
                side_effect=lambda *_args, **_kwargs: events.append("normalize"),
            ), mock.patch.object(deps, "verify_dependencies") as verify:
                with self.assertRaises(subprocess.CalledProcessError):
                    deps.install_dependencies(
                        python,
                        plan,
                        root / "cache",
                        runner=runner,
                        log=lambda _message: None,
                    )
        self.assertEqual(events, ["torch", "normalize", "pip-check"])
        verify.assert_not_called()


class LockAndStateTests(unittest.TestCase):
    def test_dependency_lock_is_partitioned_by_python_abi(self) -> None:
        payload = context(arch="arm64", sm=120, cuda=128)
        cp311 = deps.select_dependency_plan(
            payload,
            "portable",
            interpreter_fingerprint=python_fingerprint(11, arch="aarch64"),
        )
        cp312 = deps.select_dependency_plan(
            payload,
            "portable",
            interpreter_fingerprint=python_fingerprint(12, arch="aarch64"),
        )

        self.assertEqual(cp311.python_abi.lane, "cp311")
        self.assertEqual(cp312.python_abi.lane, "cp312")
        self.assertEqual(
            deps.dependency_lock_payload(cp311)["plan"]["python"]["version"],
            [3, 11],
        )
        self.assertEqual(
            deps.dependency_lock_payload(cp312)["plan"]["python"]["cacheTag"],
            "cpython-312",
        )
        self.assertNotEqual(
            deps.dependency_lock_digest(cp311), deps.dependency_lock_digest(cp312)
        )
        cp311_lane = deps.PYTHON_REQUIREMENT_LANES["cp311"]
        cp312_lane = deps.PYTHON_REQUIREMENT_LANES["cp312"]
        for field in (
            "bootstrap",
            "base",
            "common_transitive",
            "x64_render",
            "x64_render_common",
            "linux_x64_render",
            "windows_x64_render",
            "torch_lanes",
            "torch_indexes",
            "torch_common",
            "torch_platform",
            "exact_sparse",
            "exact_native",
        ):
            with self.subTest(field=field):
                self.assertIsNot(getattr(cp311_lane, field), getattr(cp312_lane, field))
        legacy = (
            "\n".join(sorted(deps.constraint_requirements(cp311))) + "\n"
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(legacy).hexdigest(),
            "89b6129130207920f4e8a4ea49c9c0971eafff2c56d0fc55a4776028dee24629",
        )
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary).resolve()
            cp311_constraints = deps.materialize_dependency_constraints(cache, cp311)
            cp312_constraints = deps.materialize_dependency_constraints(cache, cp312)
            self.assertNotEqual(cp311_constraints, cp312_constraints)
            self.assertIn(
                deps.dependency_lock_digest(cp311).encode("ascii"),
                cp311_constraints.read_bytes(),
            )
            self.assertIn(
                deps.dependency_lock_digest(cp312).encode("ascii"),
                cp312_constraints.read_bytes(),
            )

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
        # These hashes freeze the original cp311 requirement sets.  The new
        # cp312 lane is resolved independently and must preserve the same
        # audited versions until its own metadata requires an intentional fork.
        expected_cp311 = {
            "linux-x64-cu124-portable": "15a0e9f6a237d7248d335ad016012a720cd527be51fe5ca182c17bad5741ca40",
            "linux-x64-cu124-exact": "ca1053f2ca5856caa0c4910d9423dfe652c47402c8db1beb427e8995107d6923",
            "win32-x64-cu124-portable": "297bedec9332cf2f5f63241e5f7d9d9d6a62b3ea95e07f7d181ddeaae485b129",
            "win32-x64-cu124-exact": "d4b7573a301797e76b690d2c3d97ff123ebd529f382d8beca14f64adf77048c2",
            "linux-arm64-cu126": "9e76edb3b0b594869bda25ea6f743dac5b3c6fa08136568cc84744f53f163165",
            "linux-x64-cu128": "4fb1d46987a126c04cd93b620cf7a4a4128a2fbfd7af86e3cf674faccbbdf849",
            "win32-x64-cu128": "2d40eb733066ba82b57966c275c0b811ca9714de1c0a0f5ff69145b9dc4ff16b",
            "linux-arm64-cu128": "89b6129130207920f4e8a4ea49c9c0971eafff2c56d0fc55a4776028dee24629",
        }
        expected_cp312 = {
            "linux-x64-cu124-portable": "15a0e9f6a237d7248d335ad016012a720cd527be51fe5ca182c17bad5741ca40",
            "linux-x64-cu124-exact": "ca1053f2ca5856caa0c4910d9423dfe652c47402c8db1beb427e8995107d6923",
            "win32-x64-cu124-portable": "297bedec9332cf2f5f63241e5f7d9d9d6a62b3ea95e07f7d181ddeaae485b129",
            "win32-x64-cu124-exact": "d4b7573a301797e76b690d2c3d97ff123ebd529f382d8beca14f64adf77048c2",
            "linux-arm64-cu126": "9e76edb3b0b594869bda25ea6f743dac5b3c6fa08136568cc84744f53f163165",
            "linux-x64-cu128": "4fb1d46987a126c04cd93b620cf7a4a4128a2fbfd7af86e3cf674faccbbdf849",
            "win32-x64-cu128": "2d40eb733066ba82b57966c275c0b811ca9714de1c0a0f5ff69145b9dc4ff16b",
            "linux-arm64-cu128": "89b6129130207920f4e8a4ea49c9c0971eafff2c56d0fc55a4776028dee24629",
        }

        def plans_for(minor: int) -> dict[str, deps.DependencyPlan]:
            fingerprint = python_fingerprint(minor)
            arm_fingerprint = python_fingerprint(minor, arch="aarch64")
            return {
                "linux-x64-cu124-portable": deps.select_dependency_plan(
                    context(sm=75), "portable", interpreter_fingerprint=fingerprint
                ),
                "linux-x64-cu124-exact": deps.select_dependency_plan(
                    context(sm=86),
                    "exact-upstream",
                    interpreter_fingerprint=fingerprint,
                ),
                "win32-x64-cu124-portable": deps.select_dependency_plan(
                    context(system="win32", arch="x64", sm=75),
                    "portable",
                    interpreter_fingerprint=fingerprint,
                ),
                "win32-x64-cu124-exact": deps.select_dependency_plan(
                    context(system="win32", arch="x64", sm=86),
                    "exact-upstream",
                    interpreter_fingerprint=fingerprint,
                ),
                "linux-arm64-cu126": deps.select_dependency_plan(
                    context(arch="arm64", sm=90, cuda=126),
                    "portable",
                    interpreter_fingerprint=arm_fingerprint,
                ),
                "linux-x64-cu128": deps.select_dependency_plan(
                    context(arch="x64", sm=120, cuda=128),
                    "portable",
                    interpreter_fingerprint=fingerprint,
                ),
                "win32-x64-cu128": deps.select_dependency_plan(
                    context(system="win32", arch="x64", sm=120, cuda=128),
                    "portable",
                    interpreter_fingerprint=fingerprint,
                ),
                "linux-arm64-cu128": deps.select_dependency_plan(
                    context(arch="arm64", sm=120, cuda=128),
                    "portable",
                    interpreter_fingerprint=arm_fingerprint,
                ),
            }

        by_python = {minor: plans_for(minor) for minor in (11, 12)}
        expected_by_python = {11: expected_cp311, 12: expected_cp312}
        for minor, plans in by_python.items():
            for name, plan in plans.items():
                with self.subTest(python=f"3.{minor}", plan=name):
                    serialized = (
                        "\n".join(sorted(deps.constraint_requirements(plan))) + "\n"
                    ).encode("utf-8")
                    self.assertEqual(
                        hashlib.sha256(serialized).hexdigest(),
                        expected_by_python[minor][name],
                    )
                    self.assertEqual(plan.python_abi.lane, f"cp3{minor}")
        for name in expected_cp311:
            self.assertNotEqual(
                deps.dependency_lock_digest(by_python[11][name]),
                deps.dependency_lock_digest(by_python[12][name]),
            )

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
            python = fake_venv_python(root)
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
                python,
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
                python, ["check"], alias_env
            )

        self.assertEqual(command[1:3], ["-m", "pip"])
        self.assertEqual(command[0], str(python))
        self.assertIn("--isolated", command)
        self.assertIn("--disable-pip-version-check", command)
        self.assertIn("--no-input", command)
        cache_position = command.index("--cache-dir")
        self.assertEqual(command[cache_position + 1], str(cache))
        self.assertNotIn("https://evil.invalid/simple", command)
        self.assertNotIn("--cache-dir", alias_command)

    @unittest.skipIf(os.name == "nt", "POSIX venv executables are symlinks")
    def test_install_keeps_staging_venv_symlink_as_pip_executable(self) -> None:
        plan = deps.select_dependency_plan(
            context(arch="arm64", sm=90, cuda=126), "portable"
        )
        commands: list[list[str]] = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python = fake_venv_python(root, symlink=True)
            base_python = python.resolve(strict=True)
            cache = root / "cache"
            with mock.patch.object(
                deps, "_assert_target_python"
            ), mock.patch.object(deps, "_validate_linux_glibc"), mock.patch.object(
                deps, "verify_dependencies", return_value={"ok": True}
            ):
                deps.install_dependencies(
                    python,
                    plan,
                    cache,
                    runner=lambda command, _env: commands.append(list(command)),
                    log=lambda _message: None,
                )

        self.assertTrue(commands)
        self.assertTrue(all(command[0] == str(python) for command in commands))
        self.assertTrue(all(command[0] != str(base_python) for command in commands))

    def test_pip_rejects_python_outside_a_regular_venv_container(self) -> None:
        with self.assertRaises(deps.DependencyError) as system_python:
            deps.isolated_pip_command(Path(sys.executable), ["check"], {})
        self.assertEqual(system_python.exception.code, "PYTHON_PATH_INVALID")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python = fake_venv_python(root)
            alias = root / "aliased-venv"
            try:
                alias.symlink_to(python.parent.parent, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks unavailable: {exc}")
            with self.assertRaises(deps.DependencyError) as aliased:
                deps.isolated_pip_command(alias / "bin/python", ["check"], {})
        self.assertEqual(aliased.exception.code, "PYTHON_PATH_INVALID")

    def test_install_falls_back_to_owned_pip_cache_without_following_alias(self) -> None:
        plan = deps.select_dependency_plan(
            context(arch="arm64", sm=90, cuda=126), "portable"
        )
        commands: list[list[str]] = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python = fake_venv_python(root)
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
                    python,
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
    def test_python_abi_accepts_supported_release_fingerprints(self) -> None:
        fingerprints = [
            python_fingerprint(minor, arch=arch)
            for minor in (11, 12)
            for arch in ("x86_64", "aarch64")
        ]
        fingerprints.extend(
            {
                "implementation": "cpython",
                "version": [3, minor],
                "cache_tag": f"cpython-3{minor}",
                "abiflags": "",
                "soabi": f"cp3{minor}-win_amd64",
                "platform": "win-amd64",
                "machine": "amd64",
                "pointer_bits": 64,
            }
            for minor in (11, 12)
        )

        for fingerprint in fingerprints:
            with self.subTest(fingerprint=fingerprint):
                abi = deps.python_abi_from_fingerprint(fingerprint)
                self.assertEqual(abi.version, tuple(fingerprint["version"]))

    def test_python_abi_rejects_debug_and_incoherent_soabi(self) -> None:
        release = python_fingerprint(12, arch="aarch64")
        unsupported = (
            {
                **release,
                "abiflags": "d",
                "soabi": "cpython-312d-aarch64-linux-gnu",
            },
            {**release, "soabi": "not-a-python-abi"},
        )
        for fingerprint in unsupported:
            with self.subTest(fingerprint=fingerprint):
                with self.assertRaises(deps.DependencyError) as raised:
                    deps.python_abi_from_fingerprint(fingerprint)
                self.assertEqual(raised.exception.code, "PYTHON_ABI_UNSUPPORTED")

    def test_target_python_accepts_only_matching_supported_cpython_abis(self) -> None:
        base = {
            "implementation": "cpython",
            "abiflags": "",
            "platform": "linux-aarch64",
            "machine": "aarch64",
            "pointer_bits": 64,
        }
        fingerprints = (
            {
                **base,
                "version": [3, 11],
                "cache_tag": "cpython-311",
                "soabi": "cpython-311-aarch64-linux-gnu",
            },
            {
                **base,
                "version": [3, 12],
                "cache_tag": "cpython-312",
                "soabi": "cpython-312-aarch64-linux-gnu",
            },
        )
        for fingerprint in fingerprints:
            expected = deps.python_abi_from_fingerprint(fingerprint)
            completed = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=json.dumps(fingerprint), stderr=""
            )
            with self.subTest(version=fingerprint["version"]), mock.patch.object(
                deps.subprocess, "run", return_value=completed
            ):
                self.assertEqual(deps._assert_target_python(Path("/python"), expected), expected)

        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(fingerprints[1]), stderr=""
        )
        with mock.patch.object(deps.subprocess, "run", return_value=completed):
            with self.assertRaises(deps.DependencyError) as raised:
                deps._assert_target_python(
                    Path("/python"), deps.python_abi_from_fingerprint(fingerprints[0])
                )
        self.assertEqual(raised.exception.code, "PYTHON_ABI_MISMATCH")

        unsupported = {
            **base,
            "version": [3, 13],
            "cache_tag": "cpython-313",
            "soabi": "cpython-313-aarch64-linux-gnu",
        }
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(unsupported), stderr=""
        )
        with mock.patch.object(deps.subprocess, "run", return_value=completed):
            with self.assertRaises(deps.DependencyError) as raised:
                deps._assert_target_python(Path("/python"))
        self.assertEqual(raised.exception.code, "PYTHON_ABI_UNSUPPORTED")

    def test_exact_install_uses_only_the_selected_python_lane(self) -> None:
        for minor in (11, 12):
            lane_name = f"cp3{minor}"
            marker_version = f"91.{minor}.0"
            original_lane = deps.PYTHON_REQUIREMENT_LANES[lane_name]
            replacement_by_name = {
                "torch-scatter": f"torch-scatter=={marker_version}",
                "xformers": f"xformers=={marker_version}",
                "triton": f"triton=={marker_version}",
                "flash-attn": f"flash-attn=={marker_version}",
            }
            lane = replace(
                original_lane,
                exact_sparse=(
                    f"cumm-cu124=={marker_version}",
                    f"spconv-cu124=={marker_version}",
                ),
                exact_native=tuple(
                    (
                        scope,
                        replacement_by_name.get(
                            deps._requirement_name(requirement), requirement
                        ),
                    )
                    for scope, requirement in original_lane.exact_native
                ),
                torch_platform=tuple(
                    (
                        key,
                        tuple(
                            replacement_by_name.get(
                                deps._requirement_name(requirement), requirement
                            )
                            for requirement in requirements
                        ),
                    )
                    for key, requirements in original_lane.torch_platform
                ),
                torch_scatter_links=f"https://lane.invalid/{lane_name}",
            )
            commands: list[list[str]] = []

            def runner(command, env):
                del env
                commands.append(list(command))

            with self.subTest(lane=lane_name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                python = fake_venv_python(root)
                cache = root / "cache"
                native = cache / "native"
                fake_sources = deps.NativeSources(
                    native,
                    native / "nvdiffrast",
                    native / "CuMesh",
                    native / "FlexGEMM",
                    native / "o-voxel",
                )
                with mock.patch.dict(
                    deps.PYTHON_REQUIREMENT_LANES, {lane_name: lane}
                ), mock.patch.object(deps, "_assert_target_python"), mock.patch.object(
                    deps, "_validate_linux_glibc"
                ), mock.patch.object(
                    deps,
                    "native_build_environment",
                    return_value={"PATH": "/usr/bin:/bin"},
                ), mock.patch.object(
                    deps, "prepare_native_sources", return_value=fake_sources
                ), mock.patch.object(
                    deps,
                    "prepare_build_workspace",
                    side_effect=lambda source, _cache, _plan, _purpose: source,
                ), mock.patch.object(
                    deps, "verify_dependencies", return_value={"ok": True}
                ):
                    plan = deps.select_dependency_plan(
                        context(sm=86),
                        "exact-upstream",
                        interpreter_fingerprint=python_fingerprint(minor),
                    )
                    deps.install_dependencies(
                        python, plan, cache, runner=runner, log=lambda _: None
                    )

            flat = "\n".join(" ".join(command) for command in commands)
            for expected in (
                *lane.exact_sparse,
                *replacement_by_name.values(),
                lane.torch_scatter_links,
            ):
                self.assertIn(expected, flat)
            for leaked in (
                *deps.EXACT_SPARSE_REQUIREMENTS,
                deps.TORCH_SCATTER_REQUIREMENT,
                deps.XFORMERS_REQUIREMENT,
                deps.LINUX_TRITON_REQUIREMENT,
                deps.FLASH_ATTN_REQUIREMENT,
                deps.TORCH_SCATTER_LINKS,
            ):
                self.assertNotIn(leaked, flat)

    def test_exact_install_has_every_native_stage_and_no_git_command(self) -> None:
        plan = deps.select_dependency_plan(context(sm=86), "exact-upstream")
        commands: list[list[str]] = []
        environments: list[dict[str, str]] = []

        def runner(command, env):
            commands.append(list(command))
            environments.append(dict(env or {}))

        inherited = {
            "PATH": "/usr/bin:/bin",
            "HTTPS_PROXY": "http://proxy.invalid:8080",
            "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
            "PIP_INDEX_URL": "https://evil.invalid/simple",
            "PIP_CONFIG_FILE": "/tmp/evil-pip.conf",
            "PYTHONPATH": "/tmp/injected",
            "HF_TOKEN": "secret",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python = fake_venv_python(root)
            cache = root / "cache"
            native = cache / "native"
            fake_sources = deps.NativeSources(
                native,
                native / "nvdiffrast",
                native / "CuMesh",
                native / "FlexGEMM",
                native / "o-voxel",
            )
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
                    python, plan, cache, runner=runner, log=lambda _: None
                )
        flat = "\n".join(" ".join(command) for command in commands)
        for required in (
            "spconv-cu124==2.3.8",
            "torch-scatter==2.1.2+pt26cu124",
            "xformers==0.0.29.post2",
            "flash-attn==2.7.4.post1",
            "filelock==3.20.0",
            str(fake_sources.nvdiffrast),
            str(fake_sources.cumesh),
            str(fake_sources.flexgemm),
            str(fake_sources.ovoxel),
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
                    str(fake_sources.nvdiffrast),
                    str(fake_sources.cumesh),
                    str(fake_sources.flexgemm),
                    str(fake_sources.ovoxel),
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

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python = fake_venv_python(root, symlink=True)
            base_python = python.resolve(strict=True)
            with mock.patch.object(
                deps, "cpu_build_environment", return_value={"PATH": ""}
            ), mock.patch.object(deps.subprocess, "run", side_effect=fake_run):
                result = deps.verify_portable_cpu_extension(
                    python, plan, root / "cache"
                )
        self.assertEqual(result["voxelCount"], 12)
        self.assertEqual([command[0] for command in calls], [str(python), str(python)])
        self.assertNotIn(str(base_python), [command[0] for command in calls])
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
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python = fake_venv_python(root, symlink=True)
            base_python = python.resolve(strict=True)
            with mock.patch.object(
                deps.subprocess, "run", return_value=completed
            ) as run:
                deps.verify_dependencies(python, plan, root / "cache")
        smoke_command = run.call_args.args[0]
        self.assertEqual(smoke_command[0], str(python))
        self.assertNotEqual(smoke_command[0], str(base_python))
        smoke_script = smoke_command[-1]
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
