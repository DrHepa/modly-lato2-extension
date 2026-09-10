from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from lato2_modly import python_abi as abi
from _python_abi_fixtures import linux_fingerprint, windows_fingerprint


class PythonABIRegressionTests(unittest.TestCase):
    def assert_rejected(self, fingerprint, reason):
        with self.assertRaises(abi.PythonABIError) as raised:
            abi.normalize_python_abi(fingerprint)
        self.assertIn(reason, str(raised.exception))

    def test_windows_3119_and_312_accept_absent_soabi(self):
        for minor in (11, 12):
            for missing in (None, ""):
                with self.subTest(minor=minor, soabi=missing):
                    fingerprint = windows_fingerprint(minor)
                    fingerprint["soabi"] = missing
                    result = abi.normalize_python_abi(fingerprint)
                    self.assertEqual(result["soabi"], f"cp3{minor}-win_amd64")
                    self.assertEqual(result["version"], (3, minor))
                    self.assertEqual(fingerprint["soabi"], missing)

    def test_windows_key_absent_also_uses_exact_suffix(self):
        fingerprint = windows_fingerprint()
        del fingerprint["soabi"]
        self.assertEqual(abi.normalize_python_abi(fingerprint)["soabi"], "cp311-win_amd64")

    def test_explicit_and_fallback_soabi_have_same_canonical_identity(self):
        for minor in (11, 12):
            fingerprint = windows_fingerprint(minor)
            explicit = dict(fingerprint, soabi=f"cp3{minor}-win_amd64")
            self.assertEqual(abi.normalize_python_abi(fingerprint), abi.normalize_python_abi(explicit))

    def test_legacy_explicit_soabi_remains_valid(self):
        for minor in (11, 12):
            for fingerprint in (windows_fingerprint(minor), linux_fingerprint(minor), linux_fingerprint(minor, "aarch64")):
                canonical = abi.normalize_python_abi(fingerprint)
                legacy = dict(canonical, version=list(canonical["version"]))
                self.assertEqual(abi.normalize_python_abi(legacy), canonical)

    def test_linux_x64_and_arm64_preserve_release_abis(self):
        for minor in (11, 12):
            for arch in ("x86_64", "aarch64"):
                with self.subTest(minor=minor, arch=arch):
                    fingerprint = linux_fingerprint(minor, arch)
                    self.assertEqual(abi.normalize_python_abi(fingerprint)["soabi"], fingerprint["soabi"])

    def test_cp311_and_cp312_identities_stay_distinct(self):
        self.assertNotEqual(abi.normalize_python_abi(windows_fingerprint(11)), abi.normalize_python_abi(windows_fingerprint(12)))

    def test_patch_version_is_not_part_of_canonical_abi(self):
        original = windows_fingerprint()
        other_patch = dict(original, version_full=[3, 11, 99])
        self.assertEqual(abi.normalize_python_abi(original), abi.normalize_python_abi(other_patch))

    def test_canonical_payload_contains_no_executable_or_probe_only_fields(self):
        normalized = abi.normalize_python_abi(windows_fingerprint())
        self.assertEqual(set(normalized), {"implementation", "version", "cache_tag", "abiflags", "soabi", "platform", "machine", "pointer_bits"})

    def test_casing_of_platform_and_machine_is_normalized(self):
        fingerprint = dict(windows_fingerprint(), platform="WIN-AMD64", machine="AMD64")
        self.assertEqual(abi.normalize_python_abi(fingerprint), abi.normalize_python_abi(windows_fingerprint()))

    def test_invalid_versions_are_rejected(self):
        for value in (None, "3.11.9", [3], [3, 11, 9], [3, True], [3.0, 11], [3, 10], [3, 13], [3, 14]):
            with self.subTest(version=value):
                self.assert_rejected(dict(windows_fingerprint(), version=value), "version")

    def test_non_cpython_and_wrong_pointer_width_are_rejected(self):
        self.assert_rejected(dict(windows_fingerprint(), implementation="pypy"), "implementation")
        for bits in (32, 128, True, "64", 64.0, None):
            self.assert_rejected(dict(windows_fingerprint(), pointer_bits=bits), "pointer width")

    def test_cache_tag_and_abi_flags_are_checked(self):
        for minor in (11, 12):
            self.assert_rejected(dict(windows_fingerprint(minor), cache_tag="cpython-310"), "cache_tag")
            for flags in ("d", "t", "m"):
                self.assert_rejected(dict(windows_fingerprint(minor), abiflags=flags), "ABI flags")

    def test_debug_or_malformed_debug_evidence_is_rejected(self):
        for value in (True, 1, 0, "false", None):
            self.assert_rejected(dict(windows_fingerprint(), is_debug=value), "debug-build")

    def test_missing_soabi_does_not_admit_generic_or_missing_pyd_suffix(self):
        for minor in (11, 12):
            for suffix in (None, "", ".pyd"):
                with self.subTest(minor=minor, suffix=suffix):
                    self.assert_rejected(dict(windows_fingerprint(minor), ext_suffix=suffix), "EXT_SUFFIX")

    def test_wrong_version_architecture_and_debug_suffixes_are_rejected(self):
        for suffix in (".cp310-win_amd64.pyd", ".cp312-win_amd64.pyd", ".cp311-win_arm64.pyd", ".cp311-win32.pyd", "_d.cp311-win_amd64.pyd", "_d.pyd", ".cp311-win_amd64.pyd.exe", ".cpython-311-x86_64-linux-gnu.so"):
            self.assert_rejected(dict(windows_fingerprint(), ext_suffix=suffix), "EXT_SUFFIX")

    def test_explicit_wrong_soabi_is_not_hidden_by_valid_suffix(self):
        for soabi in ("cp312-win_amd64", "cpython-311-x86_64-linux-gnu", "cp311d-win_amd64", "invented"):
            self.assert_rejected(dict(windows_fingerprint(), soabi=soabi), "SOABI")

    def test_linux_still_requires_soabi(self):
        for minor in (11, 12):
            for arch in ("x86_64", "aarch64"):
                for missing in (None, ""):
                    self.assert_rejected(dict(linux_fingerprint(minor, arch), soabi=missing), "SOABI is required")

    def test_unsupported_platforms_and_inconsistent_machine_are_rejected(self):
        for platform, machine in (("win32", "x86"), ("win-arm64", "arm64"), ("win-amd64", "arm64"), ("darwin", "arm64"), ("linux-x86_64", "aarch64"), ("linux-aarch64", "x86_64"), ("linux-riscv64", "riscv64"), ("", "")):
            self.assert_rejected(dict(windows_fingerprint(), platform=platform, machine=machine), "platform/machine")

    def test_linux_musl_debug_and_wrong_arch_suffixes_are_rejected(self):
        for suffix in (".cpython-311-x86_64-linux-musl.so", ".cpython-311d-x86_64-linux-gnu.so", ".cpython-311-aarch64-linux-gnu.so"):
            self.assert_rejected(dict(linux_fingerprint(), ext_suffix=suffix), "EXT_SUFFIX")

    def test_loader_must_prefer_the_release_suffix(self):
        good = windows_fingerprint()["ext_suffix"]
        for suffixes in (None, [], good, ["_d.cp311-win_amd64.pyd", good], [".pyd", good], [good, 7]):
            self.assert_rejected(dict(windows_fingerprint(), extension_suffixes=suffixes), "loader")

    def test_non_object_metadata_has_controlled_error(self):
        for fingerprint in (None, [], "not-json", 64):
            self.assert_rejected(fingerprint, "object")

    def test_diagnostic_omits_unrelated_keys_and_escapes_newlines(self):
        fingerprint = dict(windows_fingerprint(), HF_TOKEN="do-not-print", python_exe="other-executable", machine="bad\nINJECTED")
        shown = abi.python_abi_diagnostic(fingerprint)
        self.assertNotIn("do-not-print", shown)
        self.assertNotIn("other-executable", shown)
        self.assertNotIn("\n", shown)
        self.assertEqual(json.loads(shown)["machine"], "bad\nINJECTED")

    def test_failure_includes_cause_and_actual_soabi_suffix(self):
        fingerprint = dict(windows_fingerprint(), ext_suffix="_d.pyd")
        with self.assertRaises(abi.PythonABIError) as raised:
            abi.normalize_python_abi(fingerprint)
        message = str(raised.exception)
        self.assertIn("EXT_SUFFIX", message)
        self.assertIn('"soabi": null', message)
        self.assertIn('"ext_suffix": "_d.pyd"', message)
        self.assertIn('"version_full": [3, 11, 9]', message)

    def test_real_isolated_probe_matches_in_process_collector(self):
        with tempfile.TemporaryDirectory() as directory:
            # Probe must not depend on project cwd, package imports, or pip.
            result = subprocess.run([sys.executable, "-I", "-S", "-c", abi.INTERPRETER_PROBE], cwd=directory, check=True, capture_output=True, text=True, timeout=30)
        self.assertEqual(json.loads(result.stdout), abi.current_python_fingerprint())

    def test_real_host_is_accepted_only_when_it_is_a_supported_abi(self):
        fingerprint = abi.current_python_fingerprint()
        key = (tuple(fingerprint["version"]), fingerprint["platform"], fingerprint["machine"])
        if key not in abi.SUPPORTED_RELEASE_ABIS or fingerprint["is_debug"] or fingerprint["pointer_bits"] != 64:
            self.assert_rejected(fingerprint, "supports only")
        else:
            self.assertEqual(abi.normalize_python_abi(fingerprint)["version"], tuple(sys.version_info[:2]))


if __name__ == "__main__":
    unittest.main()
