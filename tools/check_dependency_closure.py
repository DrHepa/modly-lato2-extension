"""Install the complete portable dependency graph in a disposable CI venv.

Unlike the earlier wheel-only consumer, this uses install_dependencies (including
Open3D on x64), actual package metadata/imports, pip check and the real binary
operator. GPU/native/model acceptance is explicitly NOT performed here.
"""
from __future__ import annotations

import argparse
import contextlib
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
from lato2_modly import dependencies as deps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix or Path(sys.prefix).name != 'venv.__modly_staging':
        raise RuntimeError('Use a disposable venv named venv.__modly_staging')
    system = 'win32' if os.name == 'nt' else 'linux'
    arch = 'arm64' if platform.machine().lower() in {'aarch64', 'arm64'} else 'x64'
    sm = 120 if args.torch.endswith('cu128') else 90 if arch == 'arm64' else 75
    plan = deps.select_dependency_plan(dict(platform=system, arch=arch, accelerator='cuda',
        gpu_sm=sm, cuda_version=128), 'portable')
    if plan.torch_requirements[0] != 'torch==' + args.torch:
        raise RuntimeError('Unexpected test lane')
    spec = importlib.util.spec_from_file_location('closure_setup', ROOT / 'setup.py')
    setup = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = setup
    spec.loader.exec_module(setup)
    python = Path(sys.executable)
    os.environ['PATH'] = str(python.parent)
    if os.name == 'nt':
        os.environ['PATH'] += os.pathsep + str(Path(os.environ['SYSTEMROOT']) / 'System32')
    for key in list(os.environ):
        if key.upper() in {'CC', 'CXX', 'INCLUDE', 'LIB', 'LIBPATH', 'CUDA_HOME', 'CUDA_PATH',
                           'VCINSTALLDIR', 'VSINSTALLDIR', 'VCTOOLSINSTALLDIR'}:
            del os.environ[key]
    report = {'scope': 'complete portable dependency install/imports + CPU operator; no GPU/model inference',
              'cuda_tested': False, 'system': system, 'arch': arch,
              'python': list(sys.version_info[:3]), 'raw_machine': platform.machine(), 'torch': args.torch}
    try:
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            cache = Path(tmp).resolve()
            for helper in ('cpu_build_environment', 'native_build_environment', '_msvc_environment', '_linux_cxx_environment'):
                stack.enter_context(patch.object(deps, helper, side_effect=AssertionError('compiler discovery forbidden')))
            # This local CI-only substitution stops before GPU execution. It is
            # not an environment flag, and is never used by production setup.
            with patch.object(deps, 'verify_dependencies', side_effect=deps.verify_dependency_imports):
                constraints = deps.install_dependencies(python, plan, cache, log=print)
            report['imports'] = deps.verify_dependency_imports(python, plan, cache)
            old_script = '''import importlib.metadata, json, os
expected=json.loads(os.environ["MODLY_LATO2_EXPECTED_DISTS"])
for name,wanted in expected.items():
    actual=importlib.metadata.version(name)
    if actual != wanted: raise RuntimeError(f"{name}: {actual} != {wanted}")
'''
            env = deps._dependency_probe_environment(python, plan, cache)
            old = subprocess.run([str(python), '-I', '-X', 'utf8', '-c', old_script],
                capture_output=True, text=True, encoding='utf-8', errors='replace', env=env, timeout=60)
            report['old_unconditional_metadata_exit'] = old.returncode
            report['old_unconditional_metadata_tail'] = old.stderr.splitlines()[-1:]
            if system == 'win32' and report['raw_machine'] == 'AMD64' and 'hf-xet' in report['imports']['conditional_absent']:
                assert old.returncode != 0 and 'hf-xet' in old.stderr
            binary = setup._prepare_cpu_operator(plan, cache, cache)
            setup._install_cpu_operator(python, plan, cache, binary, constraints)
            report['operator'] = deps.verify_portable_cpu_extension(python, plan, cache)
            report['result'] = 'passed'
    except Exception as exc:
        report['result'] = 'failed'
        report['error'] = getattr(exc, 'public_message', type(exc).__name__)
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
