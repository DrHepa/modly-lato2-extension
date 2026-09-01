from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from zipfile import ZipFile


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
UPSTREAM_ROOT = WORKSPACE_ROOT / "sources" / "LATO.2-upstream"

sys.path.insert(0, str(REPO_ROOT))

from lato2_modly.portable import (  # noqa: E402
    PATCHES,
    materialize_portable_runtime,
    validate_portable_runtime,
)
from lato2_modly.ovoxel_cpu import (  # noqa: E402
    EIGEN_TREE_SHA256,
    FLEXIBLE_DUAL_GRID_SHA256,
    LICENSE_SOURCE_SPECS,
    OVOXEL_CPU_BUILD_IDENTITY,
    OVOXEL_CPU_VERSION,
    materialize_ovoxel_cpu_build,
    validate_ovoxel_cpu_build,
)
from lato2_modly import ovoxel_cpu as ovoxel_cpu_module  # noqa: E402
from lato2_modly.integrity import inventory_tree, is_alias  # noqa: E402


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PortableMaterializationTests(unittest.TestCase):
    def test_patch_targets_are_unique(self):
        targets = [relative for relative, _ in PATCHES]
        self.assertEqual(len(targets), len(set(targets)))

    def test_windows_reparse_attribute_is_treated_as_alias(self):
        info = mock.Mock(st_mode=stat.S_IFREG, st_file_attributes=0x400)
        self.assertTrue(is_alias(info))

    @unittest.skipUnless(UPSTREAM_ROOT.is_dir(), "pinned LATO.2 fixture unavailable")
    def test_materialization_is_non_mutating_and_idempotent(self):
        watched = UPSTREAM_ROOT / "scripts" / "e2e_inference.py"
        before = _digest(watched)
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "LATO.2-portable"
            first = materialize_portable_runtime(UPSTREAM_ROOT, destination)
            self.assertFalse(first.reused)
            self.assertTrue(first.portable_precision_env)
            self.assertTrue(
                validate_portable_runtime(destination, upstream_root=UPSTREAM_ROOT)
            )
            second = materialize_portable_runtime(UPSTREAM_ROOT, destination)
            self.assertTrue(second.reused)
            self.assertEqual(before, _digest(watched))

            for script_name in (
                "e2e_inference.py",
                "vflow_inference.py",
                "vvae_inference.py",
                "tflow_inference.py",
            ):
                script = (destination / "scripts" / script_name).read_text("utf-8")
                self.assertIn("autocast_context(device)", script)
                self.assertNotIn(
                    'torch.autocast("cuda", dtype=torch.bfloat16)', script
                )
                self.assertNotIn("--precision", script)

    @unittest.skipUnless(UPSTREAM_ROOT.is_dir(), "pinned LATO.2 fixture unavailable")
    def test_corrupt_generated_file_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "LATO.2-portable"
            materialize_portable_runtime(UPSTREAM_ROOT, destination)
            # This is an unpatched upstream module: full-tree hashing must
            # detect corruption outside the explicit overlay list too.
            target = destination / "models" / "flow_sampler.py"
            target.write_text("corrupt\n", encoding="utf-8")
            self.assertFalse(
                validate_portable_runtime(destination, upstream_root=UPSTREAM_ROOT)
            )
            report = materialize_portable_runtime(UPSTREAM_ROOT, destination)
            self.assertFalse(report.reused)
            self.assertTrue(
                validate_portable_runtime(destination, upstream_root=UPSTREAM_ROOT)
            )
            (destination / "unexpected.py").write_text("value = 1\n", "utf-8")
            self.assertFalse(
                validate_portable_runtime(destination, upstream_root=UPSTREAM_ROOT)
            )

    @unittest.skipUnless(UPSTREAM_ROOT.is_dir(), "pinned LATO.2 fixture unavailable")
    def test_injected_native_binary_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "LATO.2-portable"
            materialize_portable_runtime(UPSTREAM_ROOT, destination)
            (destination / "modules" / "injected.pyd").write_bytes(b"native payload")
            self.assertFalse(
                validate_portable_runtime(destination, upstream_root=UPSTREAM_ROOT)
            )

    @unittest.skipUnless(UPSTREAM_ROOT.is_dir(), "pinned LATO.2 fixture unavailable")
    def test_symlink_in_generated_tree_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "LATO.2-portable"
            materialize_portable_runtime(UPSTREAM_ROOT, destination)
            try:
                (destination / "unexpected-link").symlink_to(
                    destination / "LICENSE"
                )
            except OSError:
                self.skipTest("host does not permit symlink creation")
            self.assertFalse(
                validate_portable_runtime(destination, upstream_root=UPSTREAM_ROOT)
            )

    @unittest.skipUnless(UPSTREAM_ROOT.is_dir(), "pinned LATO.2 fixture unavailable")
    def test_runtime_default_uses_verified_sibling_exact_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            source_parent = Path(temp) / "source"
            source_parent.mkdir()
            exact = source_parent / "LATO.2"
            shutil.copytree(
                UPSTREAM_ROOT,
                exact,
                ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
            )
            portable = source_parent / "LATO.2-portable"
            materialize_portable_runtime(exact, portable)
            self.assertTrue(validate_portable_runtime(portable))

    @unittest.skipUnless(UPSTREAM_ROOT.is_dir(), "pinned LATO.2 fixture unavailable")
    def test_generated_marker_cannot_self_attest_modified_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "LATO.2-portable"
            materialize_portable_runtime(UPSTREAM_ROOT, destination)
            target = destination / "models" / "flow_sampler.py"
            target.write_text("attacker controlled\n", encoding="utf-8")

            marker_path = destination / ".modly-portable.json"
            marker = json.loads(marker_path.read_text("utf-8"))
            forged = inventory_tree(
                destination,
                ignore=lambda relative, directory: (
                    not directory and relative == ".modly-portable.json"
                ),
            )
            marker["output_sha256"] = dict(forged.files)
            marker["output_tree_sha256"] = forged.digest
            marker_path.write_text(
                json.dumps(marker, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.assertFalse(
                validate_portable_runtime(destination, upstream_root=UPSTREAM_ROOT)
            )


class PortableComponentTests(unittest.TestCase):
    def _synthetic_ovoxel_build(self, root: Path) -> tuple[Path, dict[str, str]]:
        ovoxel = root / "o-voxel"
        convert = ovoxel / "src" / "convert"
        convert.mkdir(parents=True)
        (convert / "flexible_dual_grid.cpp").write_text(
            "// metadata-only test fixture\n", encoding="utf-8"
        )
        (convert / "api.h").write_text(
            "// metadata-only test fixture\n", encoding="utf-8"
        )
        eigen = root / "eigen"
        (eigen / "Eigen").mkdir(parents=True)
        (eigen / "Eigen" / "Core").write_text("fixture core\n", encoding="utf-8")
        (eigen / "Eigen" / "Dense").write_text("fixture dense\n", encoding="utf-8")
        eigen_hash = ovoxel_cpu_module._tree_hash(eigen)
        pins = {
            "EIGEN_TREE_SHA256": eigen_hash,
            "FLEXIBLE_DUAL_GRID_SHA256": _digest(
                convert / "flexible_dual_grid.cpp"
            ),
            "CONVERT_API_SHA256": _digest(convert / "api.h"),
        }
        build = root / "build-source"
        with mock.patch.object(
            ovoxel_cpu_module,
            "_validate_inputs",
            return_value=(ovoxel, eigen_hash),
        ), mock.patch.multiple(
            ovoxel_cpu_module,
            **pins,
        ):
            materialize_ovoxel_cpu_build(ovoxel, eigen, build)
        return build, pins

    def test_materialized_ovoxel_licenses_are_pinned_and_in_wheel_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            build, _pins = self._synthetic_ovoxel_build(root)
            marker = json.loads(
                (build / ".modly-ovoxel-cpu-build.json").read_text("utf-8")
            )
            self.assertEqual(marker["version"], OVOXEL_CPU_VERSION)
            self.assertEqual(marker["build_identity"], OVOXEL_CPU_BUILD_IDENTITY)
            expected_names = set(LICENSE_SOURCE_SPECS)
            bundled = build / "LICENSES"
            self.assertEqual({path.name for path in bundled.iterdir()}, expected_names)
            for name, (relative, expected_digest) in LICENSE_SOURCE_SPECS.items():
                self.assertEqual(_digest(REPO_ROOT / relative), expected_digest)
                self.assertEqual(_digest(bundled / name), expected_digest)

            # Build an actual wheel without requiring the heavyweight Torch
            # headers: the stub BuildExtension emits only a synthetic native
            # file. Wheel metadata and license-file handling remain real
            # setuptools/wheel behavior.
            stub = root / "stub"
            (stub / "torch" / "utils").mkdir(parents=True)
            (stub / "torch" / "__init__.py").write_text("", encoding="utf-8")
            (stub / "torch" / "utils" / "__init__.py").write_text("", encoding="utf-8")
            (stub / "torch" / "utils" / "cpp_extension.py").write_text(
                "from pathlib import Path\n"
                "from setuptools import Extension\n"
                "from setuptools.command.build_ext import build_ext\n"
                "def CppExtension(name, sources, **kwargs):\n"
                "    return Extension(name, sources=[])\n"
                "class BuildExtension(build_ext):\n"
                "    def run(self):\n"
                "        for extension in self.extensions:\n"
                "            output = Path(self.get_ext_fullpath(extension.name))\n"
                "            output.parent.mkdir(parents=True, exist_ok=True)\n"
                "            output.write_bytes(b'metadata-only test extension')\n",
                encoding="utf-8",
            )
            dist = root / "dist"
            env = dict(os.environ)
            env["PYTHONPATH"] = str(stub) + (
                os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
            )
            env["PYTHONNOUSERSITE"] = "1"
            subprocess.run(
                [
                    sys.executable,
                    "setup.py",
                    "bdist_wheel",
                    "--dist-dir",
                    str(dist),
                ],
                cwd=build,
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            wheels = list(dist.glob("*.whl"))
            self.assertEqual(len(wheels), 1)
            with ZipFile(wheels[0]) as wheel:
                names = set(wheel.namelist())
                metadata_name = next(
                    name for name in names if name.endswith(".dist-info/METADATA")
                )
                metadata = wheel.read(metadata_name).decode("utf-8")
                license_members = {
                    name.rsplit("/", 1)[-1]
                    for name in names
                    if ".dist-info/licenses/LICENSES/" in name
                }
            self.assertEqual(license_members, expected_names)
            self.assertIn(f"Version: {OVOXEL_CPU_VERSION}", metadata)
            for name in expected_names:
                self.assertIn(f"License-File: LICENSES/{name}", metadata)

    def test_materialized_ovoxel_license_tamper_invalidates_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            build, pins = self._synthetic_ovoxel_build(Path(temp))
            with mock.patch.multiple(ovoxel_cpu_module, **pins):
                self.assertTrue(validate_ovoxel_cpu_build(build))
                (build / "LICENSES" / "TRELLIS.2-MIT.txt").write_text(
                    "tampered\n", encoding="utf-8"
                )
                self.assertFalse(validate_ovoxel_cpu_build(build))

    def test_ovoxel_marker_cannot_self_attest_modified_build_source(self):
        with tempfile.TemporaryDirectory() as temp:
            build, pins = self._synthetic_ovoxel_build(Path(temp))
            (build / "setup.py").write_text("attacker controlled\n", encoding="utf-8")
            marker_path = build / ".modly-ovoxel-cpu-build.json"
            marker = json.loads(marker_path.read_text("utf-8"))
            forged = ovoxel_cpu_module._build_inventory(build)
            marker["build_tree_sha256"] = forged.digest
            marker["build_source_sha256"] = dict(forged.files)
            marker_path.write_text(
                json.dumps(marker, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with mock.patch.multiple(ovoxel_cpu_module, **pins):
                self.assertFalse(validate_ovoxel_cpu_build(build))

    def test_ovoxel_template_change_invalidates_and_rebuilds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            build, pins = self._synthetic_ovoxel_build(root)
            template = root / "template"
            shutil.copytree(
                REPO_ROOT / "native" / "ovoxel_cpu_template",
                template,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            ovoxel = root / "o-voxel"
            eigen = root / "eigen"
            initial_template_hash = ovoxel_cpu_module._inventory(
                template, template=True
            ).digest
            with mock.patch.multiple(
                ovoxel_cpu_module,
                **pins,
                TEMPLATE_TREE_SHA256=initial_template_hash,
            ):
                self.assertTrue(
                    validate_ovoxel_cpu_build(build, template_root=template)
                )
                (template / "src" / "bindings.cpp").write_text(
                    "// extension update\n", encoding="utf-8"
                )
                self.assertFalse(
                    validate_ovoxel_cpu_build(build, template_root=template)
                )
                changed_template_hash = ovoxel_cpu_module._inventory(
                    template, template=True
                ).digest
                with mock.patch.multiple(
                    ovoxel_cpu_module,
                    TEMPLATE_TREE_SHA256=changed_template_hash,
                ), mock.patch.object(
                    ovoxel_cpu_module,
                    "_validate_inputs",
                    return_value=(ovoxel, pins["EIGEN_TREE_SHA256"]),
                ):
                    report = materialize_ovoxel_cpu_build(
                        ovoxel,
                        eigen,
                        build,
                        template_root=template,
                    )
                    self.assertFalse(report.reused)
                    self.assertTrue(
                        validate_ovoxel_cpu_build(build, template_root=template)
                    )

    def test_software_renderer_is_deterministic(self):
        path = (
            REPO_ROOT
            / "portable_overrides"
            / "dataset"
            / "software_renderer.py"
        )
        spec = importlib.util.spec_from_file_location("software_renderer", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        import numpy as np

        vertices = np.array(
            [[-0.4, -0.4, 0.0], [0.4, -0.4, 0.0], [0.0, 0.4, 0.0]],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        renderer = module.SoftwareWhiteModelRenderer(
            img_res=64,
            bg_color=(0, 0, 0),
            add_ground=False,
            crop_to_object=True,
        )
        first, params = renderer.render(
            vertices,
            faces,
            azimuths=[45.0],
            elevations=[30.0],
            seed=7,
        )
        second, _ = renderer.render(
            vertices,
            faces,
            azimuths=[45.0],
            elevations=[30.0],
            seed=7,
        )
        self.assertEqual(first[0].shape, (64, 64, 3))
        self.assertEqual(first[0].dtype, np.uint8)
        self.assertGreater(int(first[0].sum()), 0)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(params[0]["azimuth"], 45.0)

    def test_o_voxel_shim_imports_only_conversion_surface(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_root = Path(temp)
            runtime = temp_root / "runtime"
            source_shim = REPO_ROOT / "portable_overrides" / "o_voxel"
            shutil.copytree(source_shim, runtime / "o_voxel")
            native = runtime / "lato2_ovoxel_cpu"
            native.mkdir()
            (native / "__init__.py").write_text(
                "from . import _C\n", encoding="utf-8"
            )
            (native / "_C.py").write_text(
                "def mesh_to_flexible_dual_grid_cpu(*args, **kwargs): return 'ok'\n",
                encoding="utf-8",
            )
            (runtime / "torch.py").write_text(
                "class Tensor: pass\n"
                "def no_grad():\n"
                "    def decorate(function): return function\n"
                "    return decorate\n",
                encoding="utf-8",
            )
            (runtime / "o_voxel" / "rasterize.py").write_text(
                "raise RuntimeError('must not import rasterize')\n", encoding="utf-8"
            )
            code = (
                "import sys; "
                f"sys.path.insert(0, {str(runtime)!r}); "
                "from o_voxel.convert import mesh_to_flexible_dual_grid; "
                "assert callable(mesh_to_flexible_dual_grid); "
                "assert 'lato2_ovoxel_cpu._C' in sys.modules; "
                "assert 'o_voxel.rasterize' not in sys.modules"
            )
            subprocess.run([sys.executable, "-c", code], check=True)

    @unittest.skipUnless(
        (WORKSPACE_ROOT / "sources" / "TRELLIS2" / "o-voxel").is_dir(),
        "pinned TRELLIS.2 fixture unavailable",
    )
    def test_minimal_ovoxel_build_plan_is_pinned_and_reusable(self):
        ovoxel = WORKSPACE_ROOT / "sources" / "TRELLIS2" / "o-voxel"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            eigen = root / "eigen"
            (eigen / "Eigen").mkdir(parents=True)
            (eigen / "Eigen" / "Core").write_text("fixture core\n", "utf-8")
            (eigen / "Eigen" / "Dense").write_text("fixture dense\n", "utf-8")
            eigen_hash = ovoxel_cpu_module._tree_hash(eigen)
            build = root / "build-source"
            with mock.patch.object(
                ovoxel_cpu_module, "EIGEN_TREE_SHA256", eigen_hash
            ):
                first = materialize_ovoxel_cpu_build(ovoxel, eigen, build)
                self.assertFalse(first.reused)
                self.assertTrue(validate_ovoxel_cpu_build(build))
                self.assertEqual(
                    _digest(build / "src" / "flexible_dual_grid.cpp"),
                    FLEXIBLE_DUAL_GRID_SHA256,
                )
                self.assertIn("--no-build-isolation", first.pip_install_args)
                self.assertIn("--no-deps", first.pip_install_args)
                self.assertIn("CppExtension", (build / "setup.py").read_text("utf-8"))
                self.assertNotIn(
                    "CUDAExtension", (build / "setup.py").read_text("utf-8")
                )
                bindings = (build / "src" / "bindings.cpp").read_text("utf-8")
                self.assertEqual(bindings.count("module.def("), 1)
                self.assertIn('"mesh_to_flexible_dual_grid_cpu"', bindings)
                second = materialize_ovoxel_cpu_build(ovoxel, eigen, build)
                self.assertTrue(second.reused)

    def test_sdpa_matches_independent_dense_sequences_when_torch_exists(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable in the host test interpreter")
        path = (
            REPO_ROOT
            / "portable_overrides"
            / "modules"
            / "sparse"
            / "attention"
            / "sdpa.py"
        )
        spec = importlib.util.spec_from_file_location("portable_sdpa", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        generator = torch.Generator().manual_seed(4)
        q = torch.randn((5, 2, 8), generator=generator)
        k = torch.randn((7, 2, 8), generator=generator)
        v = torch.randn((7, 2, 6), generator=generator)
        actual = module.packed_sdpa(q, k, v, [2, 3], [4, 3])
        expected = torch.cat(
            (
                module.dense_sdpa(q[:2][None], k[:4][None], v[:4][None])[0],
                module.dense_sdpa(q[2:][None], k[4:][None], v[4:][None])[0],
            )
        )
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
