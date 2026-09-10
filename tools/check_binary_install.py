"""Exercise production binary install/verification, without model or GPU setup.

Run from a disposable venv with the exact Torch wheel and NumPy installed.
For the audited cuSPARSELt ARM64 lane use a venv named venv.__modly_staging,
matching normal setup's metadata-normalization precondition.
This is not an end-to-end Modly/CUDA/inference acceptance test.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lato2_modly import binary_wheels as binaries, dependencies as deps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch', required=True)
    parser.add_argument('--report', type=Path)
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
    # Installed binaries must work with no compiler executable discoverable.
    executable_dir = str(Path(sys.executable).parent)
    os.environ['PATH'] = os.pathsep.join([executable_dir, str(Path(os.environ.get('SYSTEMROOT', 'C:/Windows'))/'System32')]) if os.name == 'nt' else executable_dir
    for name in list(os.environ):
        if name.upper() in {'CC', 'CXX', 'INCLUDE', 'LIB', 'LIBPATH', 'CUDA_HOME', 'CUDA_PATH', 'VCINSTALLDIR', 'VSINSTALLDIR', 'VCTOOLSINSTALLDIR'}:
            os.environ.pop(name, None)
    with tempfile.TemporaryDirectory() as temporary:
        cache = Path(temporary).resolve()
        # The production path must not even *look* for a toolchain.
        with patch.object(deps, 'cpu_build_environment', side_effect=AssertionError('compiler requested')), \
             patch.object(deps, 'native_build_environment', side_effect=AssertionError('CUDA toolkit requested')), \
             patch.object(deps, '_msvc_environment', side_effect=AssertionError('MSVC requested')), \
             patch.object(deps, '_linux_cxx_environment', side_effect=AssertionError('GCC requested')):
            setup._preflight_plan(plan, cache)
            # This narrow test provisions Torch directly instead of running
            # install_dependencies (which also needs a GPU). Reproduce its
            # existing audited ARM64 metadata step before the mandatory pip
            # graph check. Do not suppress pip-check failures or invent a fix.
            if deps._cusparselt_normalization_identity(plan) is not None:
                before = subprocess.run([sys.executable, '-m', 'pip', '--isolated', 'check'], capture_output=True, text=True, timeout=120)
                print('Before existing ARM64 normalization:', before.stdout, before.stderr, flush=True)
                normalized = deps.normalize_cusparselt_metadata(Path(sys.executable), plan)
                print('Existing ARM64 metadata normalization:', json.dumps(normalized, sort_keys=True), flush=True)
            binary = setup._prepare_cpu_operator(plan, cache, cache)
            constraints = deps.materialize_dependency_constraints(cache, plan)
            setup._install_cpu_operator(Path(sys.executable), plan, cache, binary, constraints)
            verified = deps.verify_portable_cpu_extension(Path(sys.executable), plan, cache)
            # Existing binary must verify again without source, compiler or download.
            with patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('network on reuse')):
                cached = binaries.ensure_wheel(cache, binary)
                repeated = deps.verify_portable_cpu_extension(Path(sys.executable), plan, cache)
            assert verified['ok'] and repeated['ok'] and cached.is_file()
            report = {'key': binary.key, 'wheel_sha256': binary.sha256,
                      'install': 'passed', 'repeat_verification': 'passed',
                      'scope': 'production CPU binary path only; not GPU or full Modly setup'}
            if args.report is not None:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
            print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
