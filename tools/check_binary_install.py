"""Exercise production binary install/verification, without model or GPU setup.

Run from a disposable venv with the exact Torch wheel and NumPy installed.
This is not an end-to-end Modly/CUDA/inference acceptance test.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lato2_modly import binary_wheels as binaries, dependencies as deps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch', required=True)
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Use a disposable virtual environment')
    spec = importlib.util.spec_from_file_location('binary_smoke_setup', ROOT/'setup.py')
    setup = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = setup
    spec.loader.exec_module(setup)
    system = 'win32' if os.name == 'nt' else 'linux'
    arch = 'arm64' if platform.machine().lower() in {'arm64','aarch64'} else 'x64'
    sm = 120 if args.torch.endswith('cu128') else 90 if arch == 'arm64' else 75
    plan = deps.select_dependency_plan(dict(platform=system, arch=arch, gpu_sm=sm,
                                           cuda_version=128, accelerator='cuda'), 'portable')
    if plan.torch_requirements[0] != 'torch==' + args.torch:
        raise RuntimeError('Unexpected Python/Torch lane')
    with tempfile.TemporaryDirectory() as temporary:
        cache = Path(temporary).resolve()
        # The production path must not even *look* for a toolchain.
        with patch.object(deps, 'cpu_build_environment', side_effect=AssertionError('compiler requested')), \
             patch.object(deps, 'native_build_environment', side_effect=AssertionError('CUDA toolkit requested')), \
             patch.object(deps, '_msvc_environment', side_effect=AssertionError('MSVC requested')), \
             patch.object(deps, '_linux_cxx_environment', side_effect=AssertionError('GCC requested')):
            setup._preflight_plan(plan, cache)
            binary = setup._prepare_cpu_operator(plan, cache, cache)
            constraints = deps.materialize_dependency_constraints(cache, plan)
            setup._install_cpu_operator(Path(sys.executable), plan, cache, binary, constraints)
            verified = deps.verify_portable_cpu_extension(Path(sys.executable), plan, cache)
            # Existing binary must verify again without source, compiler or download.
            with patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('network on reuse')):
                cached = binaries.ensure_wheel(cache, binary)
                repeated = deps.verify_portable_cpu_extension(Path(sys.executable), plan, cache)
            assert verified['ok'] and repeated['ok'] and cached.is_file()
            print(json.dumps({'key': binary.key, 'wheel_sha256': binary.sha256,
                              'install': 'passed', 'repeat_verification': 'passed',
                              'scope': 'production CPU binary path only; not GPU or full Modly setup'}, indent=2))


if __name__ == '__main__':
    main()
