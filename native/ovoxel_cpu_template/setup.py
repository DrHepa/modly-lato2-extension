"""Build the minimal CPU-only o-voxel function used by LATO.2."""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
from wheel.bdist_wheel import bdist_wheel


ROOT = Path(__file__).resolve().parent


class LicenseLayoutBdistWheel(bdist_wheel):
    """Normalize declared licenses to the PEP 639 wheel location."""

    def egg2dist(self, egginfo_path, distinfo_path):
        super().egg2dist(egginfo_path, distinfo_path)
        distinfo = Path(distinfo_path)
        for raw_license in self.license_paths:
            source = Path(raw_license).resolve()
            try:
                relative = source.relative_to(ROOT)
            except ValueError as exc:
                raise RuntimeError("license file escaped the build root") from exc
            if len(relative.parts) != 2 or relative.parts[0] != "LICENSES":
                raise RuntimeError("license file is outside the pinned LICENSES payload")
            desired = distinfo / "licenses" / relative
            legacy = distinfo / source.name
            if desired.is_file():
                if legacy.is_file() and legacy != desired:
                    legacy.unlink()
                continue
            if not legacy.is_file():
                raise RuntimeError(f"wheel omitted declared license: {relative}")
            desired.parent.mkdir(parents=True, exist_ok=True)
            legacy.replace(desired)


if os.name == "nt":
    compile_args = ["/O2", "/std:c++17", "/EHsc"]
else:
    compile_args = ["-O3", "-std=c++17"]

setup(
    name="modly-lato2-ovoxel-cpu",
    version="0.0.1.post2",
    description="Pinned CPU flexible-dual-grid operator for Modly LATO.2",
    license_files=["LICENSES/*"],
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name="lato2_ovoxel_cpu._C",
            sources=[
                str(ROOT / "src" / "bindings.cpp"),
                str(ROOT / "src" / "flexible_dual_grid.cpp"),
            ],
            include_dirs=[
                str(ROOT / "src"),
                str(ROOT / "third_party" / "eigen"),
            ],
            extra_compile_args=compile_args,
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
        "bdist_wheel": LicenseLayoutBdistWheel,
    },
    python_requires=">=3.10",
    zip_safe=False,
)
