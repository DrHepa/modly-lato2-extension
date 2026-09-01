"""Build the minimal CPU-only o-voxel function used by LATO.2."""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


ROOT = Path(__file__).resolve().parent
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
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    zip_safe=False,
)
