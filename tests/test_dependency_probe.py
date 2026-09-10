"""Regression coverage for conditional metadata and actionable smoke failures."""
from __future__ import annotations

import contextlib
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from lato2_modly import dependencies as deps
from lato2_modly.dependency_probe import (
    CONDITIONAL_DISTRIBUTION_MARKERS, IMPORT_PROBE, METADATA_PROBE, failure_detail,
)
from _python_abi_fixtures import windows_fingerprint


class ConditionalMetadataTests(unittest.TestCase):
    def probe(self, machine, installed, expected=None):
        expected = expected or {'hf-xet': '1.6.0', 'torch': '2.6.0+cu124'}
        def version(name):
            if name not in installed:
                raise importlib.metadata.PackageNotFoundError(name)
            return installed[name]
        scope = {}
        with mock.patch('platform.machine', return_value=machine), \
             mock.patch.dict('os.environ', {
                 'MODLY_LATO2_EXPECTED_DISTS': json.dumps(expected),
                 'MODLY_LATO2_CONDITIONAL_DISTS': json.dumps(CONDITIONAL_DISTRIBUTION_MARKERS),
             }), mock.patch.object(importlib.metadata, 'version', side_effect=version), \
             contextlib.redirect_stderr(io.StringIO()):
            exec(METADATA_PROBE, scope)
        return scope

    def test_windows_uppercase_amd64_does_not_require_absent_xet(self):
        result = self.probe('AMD64', {'torch': '2.6.0+cu124'})
        self.assertEqual(result['conditional_absent'], ['hf-xet'])

    def test_old_unconditional_loop_reproduces_reported_failure(self):
        def missing(name):
            raise importlib.metadata.PackageNotFoundError(name)
        with mock.patch.object(importlib.metadata, 'version', side_effect=missing):
            with self.assertRaises(importlib.metadata.PackageNotFoundError):
                importlib.metadata.version('hf-xet')
        self.probe('AMD64', {'torch': '2.6.0+cu124'})

    def test_active_marker_still_requires_xet(self):
        for machine in ('amd64', 'x86_64', 'arm64', 'aarch64'):
            with self.subTest(machine=machine), self.assertRaises(importlib.metadata.PackageNotFoundError):
                self.probe(machine, {'torch': '2.6.0+cu124'})

    def test_installed_xet_is_always_version_checked(self):
        for machine in ('AMD64', 'x86_64', 'aarch64'):
            with self.subTest(machine=machine):
                self.probe(machine, {'hf-xet': '1.6.0', 'torch': '2.6.0+cu124'})
                with self.assertRaisesRegex(RuntimeError, 'hf-xet version'):
                    self.probe(machine, {'hf-xet': '9.0.0', 'torch': '2.6.0+cu124'})

    def test_unconditional_missing_package_still_fails(self):
        with self.assertRaises(importlib.metadata.PackageNotFoundError):
            self.probe('AMD64', {})

    def test_wrong_torch_build_still_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'torch version'):
            self.probe('AMD64', {'torch': '2.6.0+cpu'})

    def test_constraint_pin_is_retained_for_both_python_lanes(self):
        for minor in (11, 12):
            plan = deps.select_dependency_plan(
                dict(platform='win32', arch='x64', accelerator='cuda', gpu_sm=75, cuda_version=128),
                'portable', interpreter_fingerprint=windows_fingerprint(minor))
            self.assertIn('hf-xet==1.6.0', deps.constraint_requirements(plan))
            self.assertEqual(deps.dependency_lock_payload(plan)['conditionalDistributionMarkers'],
                             CONDITIONAL_DISTRIBUTION_MARKERS)


class DiagnosticTests(unittest.TestCase):
    def test_original_stderr_and_last_stage_are_visible(self):
        exc = subprocess.CalledProcessError(1, ['DO_NOT_ECHO_SCRIPT'], stderr=(
            'LATO2_PROBE_STAGE=import:numpy\nLATO2_PROBE_STAGE=import:torch\n'
            'Traceback (most recent call last):\nOSError: [WinError 126] DLL load failed\n'))
        result = failure_detail(exc)
        self.assertIn('stage=import:torch', result)
        self.assertIn('WinError 126', result)
        self.assertNotIn('DO_NOT_ECHO_SCRIPT', result)

    def test_timeout_partial_bytes_and_native_crash(self):
        exc = subprocess.TimeoutExpired('not-the-script', 3, stderr=b'LATO2_PROBE_STAGE=cuda:matmul\n')
        self.assertIn('stage=cuda:matmul; timeout', failure_detail(exc))
        exc = subprocess.CalledProcessError(3221225477, 'not-the-script', stderr='LATO2_PROBE_STAGE=import:open3d\n')
        self.assertIn('exit=3221225477', failure_detail(exc))
        self.assertIn('import:open3d', failure_detail(exc))

    def test_output_redaction_and_size_limit(self):
        exc = subprocess.CalledProcessError(1, 'ignored', stderr=(
            'x' * 30000 + '\nLATO2_PROBE_STAGE=import:torch\n'
            'Authorization: Bearer abcdefghijklmn\n'
            'https://user:secretvalue@host/asset?token=TOKENVALUE&sig=SIGNATUREVALUE\n'
            'password="PASSVALUE" hf_abcdefghijklmnopqrst\n'
            '/owned/user/extension/problem.py\nRuntimeError: useful reason'))
        result = failure_detail(exc, paths={'/owned/user': '<user>'})
        self.assertLess(len(result), 6200)
        for secret in ('secretvalue', 'TOKENVALUE', 'SIGNATUREVALUE', 'PASSVALUE',
                       'abcdefghijklmn', '/owned/user'):
            self.assertNotIn(secret, result)
        self.assertIn('RuntimeError: useful reason', result)

    def test_oserror_does_not_echo_command_or_filename(self):
        result = failure_detail(FileNotFoundError(2, 'missing', '/private/path'))
        self.assertIn('errno=2', result)
        self.assertNotIn('/private/path', result)

    def test_dependency_error_preserves_stable_code_and_cause(self):
        exc = subprocess.CalledProcessError(1, 'hidden', stderr=(
            'LATO2_PROBE_STAGE=version:hf-xet\n'
            'PackageNotFoundError: No package metadata was found for hf-xet'))
        with mock.patch.object(deps.subprocess, 'run', side_effect=exc):
            with self.assertRaises(deps.DependencyError) as caught:
                deps._run_dependency_probe(Path('/venv/bin/python'), 'hidden script', {})
        self.assertEqual(caught.exception.code, 'DEPENDENCY_SMOKE_FAILED')
        self.assertIn('version:hf-xet', caught.exception.public_message)
        self.assertIn('PackageNotFoundError', caught.exception.public_message)
        self.assertIs(caught.exception.__cause__, exc)

    def test_invalid_json_is_not_accepted(self):
        result = subprocess.CompletedProcess([], 0, stdout='not-json', stderr='')
        with mock.patch.object(deps.subprocess, 'run', return_value=result):
            with self.assertRaises(deps.DependencyError):
                deps._run_dependency_probe(Path('/venv/bin/python'), 'script', {})

    def test_real_subprocess_traceback_survives_wrapper(self):
        with self.assertRaises(deps.DependencyError) as caught:
            deps._run_dependency_probe(Path(sys.executable),
                'import sys; print("LATO2_PROBE_STAGE=import:fixture",file=sys.stderr,flush=True); raise RuntimeError("fixture-reason")', {})
        self.assertIn('fixture-reason', caught.exception.public_message)
        self.assertIn('import:fixture', caught.exception.public_message)

    def test_production_smoke_still_requires_cuda_and_capability_match(self):
        import inspect
        source = inspect.getsource(deps.verify_dependencies)
        self.assertIn('if not torch.cuda.is_available()', source)
        self.assertIn('torch.cuda.synchronize()', source)
        self.assertIn('GPU_CHANGED', source)
        self.assertNotIn('verify_dependency_imports', source)
        self.assertIn('import filelock', IMPORT_PROBE)


if __name__ == '__main__':
    unittest.main()
