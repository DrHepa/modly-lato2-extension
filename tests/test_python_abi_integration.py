"""Run against the COMPLETE patched repository, not the hotfix bundle alone.

The Windows metadata tests below are mocked. The final tests use the actual
runner's interpreter and a real temporary venv; the CI matrix supplies Windows
and Linux interpreters. No model weights, Torch, CUDA, or pip installs are used.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from lato2_modly import dependencies as deps
from lato2_modly import python_abi as abi
from _python_abi_fixtures import windows_fingerprint

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("modly_lato2_abi_setup", PROJECT / "setup.py")
assert SPEC is not None and SPEC.loader is not None
setup = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = setup
SPEC.loader.exec_module(setup)


class PythonABIIntegrationTests(unittest.TestCase):
    def test_setup_and_dependency_verification_share_probe(self):
        self.assertIs(setup.INTERPRETER_PROBE, abi.INTERPRETER_PROBE)
        self.assertIs(deps.INTERPRETER_PROBE, abi.INTERPRETER_PROBE)
        self.assertEqual(deps._current_python_fingerprint(), abi.current_python_fingerprint())

    def test_wrapper_keeps_existing_pythonabi_type_and_error_code(self):
        result = deps.python_abi_from_fingerprint(windows_fingerprint())
        self.assertIsInstance(result, deps.PythonABI)
        self.assertEqual(result.lane, "cp311")
        with self.assertRaises(deps.DependencyError) as raised:
            deps.python_abi_from_fingerprint(dict(windows_fingerprint(), ext_suffix=".pyd"))
        self.assertEqual(raised.exception.code, "PYTHON_ABI_UNSUPPORTED")
        self.assertIn("EXT_SUFFIX", raised.exception.public_message)

    def test_lock_identity_is_stable_and_python_lanes_remain_separate(self):
        context = {"platform": "win32", "arch": "x64", "gpu_sm": 75, "cuda_version": 124, "accelerator": "cuda"}
        digests = []
        for minor in (11, 12):
            raw = windows_fingerprint(minor)
            explicit = dict(raw, soabi=f"cp3{minor}-win_amd64")
            plan = deps.select_dependency_plan(context, "portable", interpreter_fingerprint=raw)
            other = deps.select_dependency_plan(context, "portable", interpreter_fingerprint=explicit)
            self.assertEqual(deps.dependency_lock_digest(plan), deps.dependency_lock_digest(other))
            self.assertEqual(plan.python_abi.lane, f"cp3{minor}")
            digests.append(deps.dependency_lock_digest(plan))
        self.assertNotEqual(*digests)

    def test_setup_accepts_windows_raw_fingerprint_and_logs_evidence(self):
        for minor in (11, 12):
            with self.subTest(minor=minor), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                payload = {"python_exe": sys.executable, "ext_dir": str(root), "platform": "win32", "arch": "x64", "gpu_sm": 75, "cuda_version": 124, "accelerator": "cuda"}
                raw = windows_fingerprint(minor)
                output = io.StringIO()
                with patch.object(setup, "current_platform_name", return_value="win32"), patch.object(setup.platform, "machine", return_value="AMD64"), patch.object(setup, "interpreter_fingerprint", return_value=raw), redirect_stdout(output):
                    context = setup.validate_context(payload, root)
                self.assertEqual(context.platform_name, "win32")
                self.assertEqual(context.arch, "x64")
                self.assertIsNone(context.host_fingerprint["soabi"])
                self.assertIn(str(Path(sys.executable).resolve()), output.getvalue().replace("\\\\", "\\"))
                self.assertIn('"soabi": null', output.getvalue())
                self.assertIn(f".cp3{minor}-win_amd64.pyd", output.getvalue())

    def test_setup_failure_contains_executable_and_specific_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            payload = {"python_exe": sys.executable, "ext_dir": str(root), "platform": sys.platform, "arch": platform.machine(), "gpu_sm": 75}
            with patch.object(setup, "interpreter_fingerprint", return_value=dict(windows_fingerprint(), ext_suffix=".pyd")), redirect_stdout(io.StringIO()):
                with self.assertRaises(setup.SetupFailure) as raised:
                    setup.validate_context(payload, root)
            self.assertEqual(raised.exception.code, "PYTHON_ABI_UNSUPPORTED")
            self.assertIn("python_exe=", raised.exception.public_message)
            self.assertIn("EXT_SUFFIX", raised.exception.public_message)

    def test_target_probe_uses_shared_script_and_sanitized_environment(self):
        raw = windows_fingerprint()
        with patch.dict(os.environ, {"PATH": os.defpath, "PYTHONPATH": "/injected", "HF_TOKEN": "not-for-child"}, clear=True), patch.object(deps.subprocess, "run", return_value=types.SimpleNamespace(stdout=json.dumps(raw))) as run:
            actual = deps._assert_target_python(Path(sys.executable))
        self.assertEqual(actual.soabi, "cp311-win_amd64")
        self.assertEqual(run.call_args.args[0][1:], ["-I", "-S", "-c", abi.INTERPRETER_PROBE])
        child_env = run.call_args.kwargs["env"]
        self.assertNotIn("PYTHONPATH", child_env)
        self.assertNotIn("HF_TOKEN", child_env)

    def test_target_rejects_other_python_lane(self):
        expected = deps.python_abi_from_fingerprint(windows_fingerprint(11))
        with patch.object(deps.subprocess, "run", return_value=types.SimpleNamespace(stdout=json.dumps(windows_fingerprint(12)))):
            with self.assertRaises(deps.DependencyError) as raised:
                deps._assert_target_python(Path(sys.executable), expected)
        self.assertEqual(raised.exception.code, "PYTHON_ABI_MISMATCH")

    def test_target_failure_keeps_code_and_includes_executable(self):
        raw = dict(windows_fingerprint(), ext_suffix="_d.pyd")
        with patch.object(deps.subprocess, "run", return_value=types.SimpleNamespace(stdout=json.dumps(raw))):
            with self.assertRaises(deps.DependencyError) as raised:
                deps._assert_target_python(Path(sys.executable))
        self.assertEqual(raised.exception.code, "PYTHON_ABI_UNSUPPORTED")
        self.assertIn("python_exe=", raised.exception.public_message)
        self.assertIn("EXT_SUFFIX", raised.exception.public_message)

    def test_real_setup_probe_matches_actual_collector(self):
        actual = setup.interpreter_fingerprint(Path(sys.executable))
        self.assertEqual(actual, deps._current_python_fingerprint())
        self.assertIn("ext_suffix", actual)
        self.assertIn("is_debug", actual)

    def test_real_temporary_venv_matches_base_python_abi(self):
        host = deps._current_python_fingerprint()
        key = (tuple(host["version"]), host["platform"], host["machine"])
        if key not in abi.SUPPORTED_RELEASE_ABIS or host["is_debug"]:
            self.skipTest("real-venv smoke requires a supported release CPython 3.11/3.12 runner")
        expected = deps.python_abi_from_fingerprint(host)
        with tempfile.TemporaryDirectory() as directory:
            venv = Path(directory) / "venv"
            subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, timeout=60, env=deps.sanitize_subprocess_environment())
            python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            self.assertEqual(setup.interpreter_fingerprint(python), host)
            self.assertEqual(deps._assert_target_python(python, expected), expected)


if __name__ == "__main__":
    unittest.main()
