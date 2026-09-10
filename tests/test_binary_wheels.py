"""Binary-only provisioning tests; synthetic wheels are never release artifacts."""
from __future__ import annotations
import base64
from dataclasses import replace
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from lato2_modly import binary_wheels as bw, dependencies as deps
from _python_abi_fixtures import windows_fingerprint, linux_fingerprint

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('binary_test_setup', ROOT/'setup.py')
setup=importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name]=setup; SPEC.loader.exec_module(setup)


def plan(minor=11, *, system='win32', arch='x64', sm=75):
    fp=windows_fingerprint(minor) if system=='win32' else linux_fingerprint(minor, arch='aarch64' if arch=='arm64' else 'x86_64')
    return deps.select_dependency_plan(dict(platform=system,arch=arch,gpu_sm=sm,cuda_version=128,accelerator='cuda'), 'portable',interpreter_fingerprint=fp)


def synthetic(root, *, minor=11):
    p=plan(minor)
    filename=f'modly_lato2_ovoxel_cpu-{bw.OVOXEL_CPU_VERSION}-1torch260cu124-cp3{minor}-cp3{minor}-win_amd64.whl'
    dist=f'modly_lato2_ovoxel_cpu-{bw.OVOXEL_CPU_VERSION}.dist-info'
    data={f'lato2_ovoxel_cpu/_C.cp3{minor}-win_amd64.pyd': b'SYNTHETIC-NOT-EXECUTABLE',
          'lato2_ovoxel_cpu/__init__.py': b'# synthetic test fixture\n',
          f'{dist}/METADATA':f'Name: {bw.OVOXEL_CPU_DISTRIBUTION}\nVersion: {bw.OVOXEL_CPU_VERSION}\n'.encode(),
          f'{dist}/WHEEL':f'Wheel-Version: 1.0\nRoot-Is-Purelib: false\nBuild: 1torch260cu124\nTag: cp3{minor}-cp3{minor}-win_amd64\n'.encode()}
    for name,(relative,_) in bw.LICENSE_SOURCE_SPECS.items():
        data[f'{dist}/licenses/LICENSES/{name}']=(ROOT/relative).read_bytes()
    files={n:hashlib.sha256(b).hexdigest() for n,b in data.items()}
    data[f'{dist}/RECORD']=''.join(f'{n},sha256={base64.urlsafe_b64encode(bytes.fromhex(files[n])).rstrip(b"=").decode()},{len(b)}\n' for n,b in data.items()).encode()+f'{dist}/RECORD,,\n'.encode()
    path=root/filename
    with ZipFile(path,'w') as archive:
        for n,b in data.items(): archive.writestr(n,b)
    value=dict(key=bw.wheel_key(p),filename=filename,release='ovoxel-cpu-test',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size=path.stat().st_size,
               build_identity=bw.OVOXEL_CPU_BUILD_IDENTITY,template_sha256=bw.TEMPLATE_TREE_SHA256,source_commit='a'*40,files=files,glibc_min=None)
    return p,bw.parse_spec(value),path


class BinaryInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.plan,self.spec,self.wheel=synthetic(self.root)

    def test_valid_wheel_checks_whole_archive_and_record(self):
        bw.verify_wheel(self.wheel,self.spec)

    def test_no_matching_wheel_does_not_build_or_download(self):
        with patch.object(bw,'ensure_asset') as fetch, patch.object(subprocess,'run') as run:
            with self.assertRaises(bw.BinaryWheelError) as raised:
                bw.select_wheel(self.plan,inventory=[])
        self.assertEqual(raised.exception.code,'BINARY_WHEEL_UNAVAILABLE'); fetch.assert_not_called(); run.assert_not_called()

    def test_python_and_torch_builds_never_share_a_binary(self):
        for p in (plan(12),plan(sm=120),plan(system='linux')):
            with self.subTest(key=bw.wheel_key(p)),self.assertRaises(bw.BinaryWheelError):
                bw.select_wheel(p,inventory=[self.spec])

    def test_preflight_never_checks_toolchains(self):
        with patch.object(bw,'select_wheel',return_value=self.spec),patch.object(deps,'cpu_build_environment',side_effect=AssertionError('compiler')),patch.object(deps,'native_build_environment',side_effect=AssertionError('nvcc')):
            setup._preflight_plan(self.plan,self.root)
            self.assertEqual(setup._prepare_cpu_operator(self.plan,self.root,self.root),self.spec)

    def test_auto_is_portable_on_windows_linux_x64_and_arm64(self):
        for system,arch in [('win32','x64'),('linux','x64'),('linux','arm64')]:
            p=plan(system=system,arch=arch,sm=86)
            context=dict(platform=system,arch=arch,gpu_sm=86,cuda_version=128,accelerator='cuda')
            fp=windows_fingerprint(11) if system=='win32' else linux_fingerprint(11,arch='aarch64' if arch=='arm64' else 'x86_64')
            self.assertFalse(deps.select_dependency_plan(context,'auto',interpreter_fingerprint=fp).install_native_stack)

    def test_exact_source_profile_remains_explicit(self):
        p=deps.select_dependency_plan(dict(platform='win32',arch='x64',gpu_sm=86,cuda_version=128,accelerator='cuda'),'exact-upstream',interpreter_fingerprint=windows_fingerprint(11))
        self.assertTrue(p.install_native_stack)

    def test_wrong_source_license_or_filename_is_rejected(self):
        for field,value in [('filename','evil.whl'),('build_identity','b'*64),('template_sha256','b'*64),('source_commit','main'),('sha256','0'*64),('release','../main'),('size',True)]:
            record=dict(self.spec.__dict__); record[field]=value
            with self.subTest(field=field),self.assertRaises(bw.BinaryWheelError): bw.parse_spec(record)
        record=dict(self.spec.__dict__,files=dict(self.spec.files)); record['files'][next(n for n in record['files'] if '/licenses/' in n)]='b'*64
        with self.assertRaises(bw.BinaryWheelError): bw.parse_spec(record)

    def test_unsafe_paths_and_path_hooks_are_rejected(self):
        for name in ('../x','/x','C:/x','a\\b','lato2_ovoxel_cpu/evil.pth'):
            record=dict(self.spec.__dict__,files={**self.spec.files,name:'a'*64})
            with self.subTest(name=name),self.assertRaises(bw.BinaryWheelError): bw.parse_spec(record)

    def test_duplicate_index_keys_are_rejected(self):
        path=self.root/'index.json'
        path.write_text(json.dumps(dict(schema=bw.SCHEMA,artifacts=[self.spec.__dict__,self.spec.__dict__])))
        with self.assertRaises(bw.BinaryWheelError): bw.load_inventory(path)

    def test_corruption_is_rejected_before_install(self):
        self.wheel.write_bytes(self.wheel.read_bytes()+b'corrupt')
        with self.assertRaises(bw.BinaryWheelError): bw.verify_wheel(self.wheel,self.spec)

    def test_valid_cache_is_used_without_network(self):
        with patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('network')):
            cache=self.root/'cache'; cache.mkdir()
            target=cache/self.spec.asset.relative_path; target.parent.mkdir(parents=True)
            target.write_bytes(self.wheel.read_bytes())
            self.assertEqual(bw.ensure_wheel(cache,self.spec,log=lambda _:None),target)

    def test_wheel_alias_and_hardlink_are_rejected(self):
        alias=self.root/'alias.whl'
        try: alias.symlink_to(self.wheel)
        except OSError: self.skipTest('symlinks unavailable')
        with self.assertRaises(bw.BinaryWheelError): bw.verify_wheel(alias,self.spec)
        alias.unlink(); os.link(self.wheel,alias)
        with self.assertRaises(bw.BinaryWheelError): bw.verify_wheel(alias,self.spec)

    def test_pip_command_is_binary_only_no_index_no_deps(self):
        venv=self.root/'venv'; (venv/'bin').mkdir(parents=True); python=venv/'bin/python'; python.touch(); (venv/'pyvenv.cfg').write_text('include-system-site-packages = false\n')
        with patch.object(bw,'ensure_wheel',return_value=self.wheel),patch.object(deps,'cpu_build_environment',side_effect=AssertionError('compiler')),patch.object(setup,'_run_checked') as run:
            setup._install_cpu_operator(python,self.plan,self.root,self.spec,self.root/'lock.txt')
        command=run.call_args.args[0]
        for option in ('--only-binary=:all:','--no-index','--no-deps','--constraint','--isolated'): self.assertIn(option,command)
        self.assertNotIn('--no-build-isolation',command)
        self.assertEqual(command[-1],str(self.wheel))

    def test_portable_never_accepts_a_source_directory(self):
        with self.assertRaises(setup.SetupFailure):
            setup._install_cpu_operator(Path(sys.executable),self.plan,self.root,SimpleNamespace(pip_install_args=()),self.root/'lock')

    def test_state_binds_exact_wheel_hash(self):
        with patch.object(bw,'select_wheel',return_value=self.spec): a=deps.dependency_state_payload(self.plan,windows_fingerprint(11))
        with patch.object(bw,'select_wheel',return_value=replace(self.spec,sha256='b'*64)): b=deps.dependency_state_payload(self.plan,windows_fingerprint(11))
        self.assertNotEqual(a,b)
        self.assertEqual(a['binaryWheel']['sha256'],self.spec.sha256)

    def test_runtime_environment_does_not_probe_a_compiler(self):
        with patch.object(deps,'_msvc_environment',side_effect=AssertionError('MSVC')),patch.object(deps,'_linux_cxx_environment',side_effect=AssertionError('CXX')),patch.object(subprocess,'run',side_effect=AssertionError('spawn')):
            self.assertEqual(deps.cpu_runtime_environment(self.plan,self.root,base_env={'PATH':'','TOKEN':'secret'}),{'PYTHONNOUSERSITE':'1'})

    def test_installed_hash_failure_is_actionable(self):
        with patch.object(bw.subprocess,'run',side_effect=subprocess.CalledProcessError(1,['python'])):
            with self.assertRaises(bw.BinaryWheelError) as raised: bw.verify_installed(Path(sys.executable),self.spec,env={})
        self.assertEqual(raised.exception.code,'BINARY_INSTALLED_INVALID')

    def test_verification_calls_installed_hash_check_before_operator(self):
        venv=self.root/'venv'; (venv/'bin').mkdir(parents=True); python=venv/'bin/python'; python.touch(); (venv/'pyvenv.cfg').write_text('include-system-site-packages = false\n')
        with patch.object(bw,'select_wheel',return_value=self.spec),patch.object(bw,'verify_installed',side_effect=bw.BinaryWheelError('BINARY_INSTALLED_INVALID','test corruption')),patch.object(deps,'cpu_build_environment',side_effect=AssertionError('compiler')),patch.object(deps.subprocess,'run') as run:
            with self.assertRaises(bw.BinaryWheelError): deps.verify_portable_cpu_extension(python,self.plan,self.root)
        run.assert_not_called()


class BinaryTransactionTests(unittest.TestCase):
    def test_binary_install_promotion_and_repair_never_require_compiler(self):
        """Exercise actual filesystem promotion; GPU/model checks are mocked."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            p, spec, wheel = synthetic(root)
            raw = windows_fingerprint(11)
            ctx = setup.SetupContext(Path(sys.executable), root, 75, 128, 'cuda',
                                     'linux', 'x64', {}, raw)
            cache = root / 'cache'; cache.mkdir()
            def make_venv(context, directory):
                python = directory / 'bin/python'
                python.parent.mkdir(parents=True); python.touch()
                (directory/'pyvenv.cfg').write_text('include-system-site-packages = false\n')
                return python
            def install_dependencies(python, plan, cache_root, **kwargs):
                return deps.materialize_dependency_constraints(cache_root, plan)
            with patch.object(setup, '_create_venv', side_effect=make_venv), \
                 patch.object(setup, '_preflight_install_storage'), \
                 patch.object(setup, 'interpreter_fingerprint', return_value=raw), \
                 patch.object(deps, 'install_dependencies', side_effect=install_dependencies) as install, \
                 patch.object(deps, 'verify_dependencies', return_value={'ok': True}), \
                 patch.object(deps, 'verify_portable_cpu_extension', return_value={'ok': True}), \
                 patch.object(bw, 'select_wheel', return_value=spec), \
                 patch.object(bw, 'ensure_wheel', return_value=wheel) as ensure, \
                 patch.object(setup, '_run_checked') as run, \
                 patch.object(deps, 'cpu_build_environment', side_effect=AssertionError('compiler')), \
                 patch.object(deps, 'native_build_environment', side_effect=AssertionError('nvcc')):
                result = setup.install_or_reuse_environment(ctx, p, cache, spec)
                self.assertFalse(result.reused)
                self.assertTrue((root/'venv/bin/python').is_file())
                self.assertFalse((root/setup.VENV_STAGING_NAME).exists())
                setup._commit_environment_promotion(ctx, result.promotion)
                install.assert_called_once(); ensure.assert_called_once()
                result2 = setup.install_or_reuse_environment(ctx, p, cache, spec)
                self.assertTrue(result2.reused)
                install.assert_called_once(); ensure.assert_called_once()
                installs = [call.args[0] for call in run.call_args_list if 'install' in call.args[0]]
                self.assertEqual(len(installs), 1)
                self.assertIn('--only-binary=:all:', installs[0])

if __name__=='__main__': unittest.main()
