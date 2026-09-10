"""Probe the actual Modly Python; optionally check a temporary isolated venv.

Run this from the complete patched repository. No packages/weights are installed.
Use --python with Modly's actual python_exe, rather than assuming PATH matches it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lato2_modly import dependencies as deps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--check-venv", action="store_true")
    args = parser.parse_args()
    python = args.python.expanduser().absolute()
    report = {"python_exe": str(python), "scope": "Python ABI only; no CUDA/inference validation"}
    try:
        spec = importlib.util.spec_from_file_location("modly_lato2_diagnostic_setup", ROOT / "setup.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("setup.py could not be loaded")
        setup = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = setup
        spec.loader.exec_module(setup)
        raw = setup.interpreter_fingerprint(python)
        report["fingerprint"] = raw
        normalized = deps.python_abi_from_fingerprint(raw)
        report["abi"] = normalized.payload()
        if args.check_venv:
            with tempfile.TemporaryDirectory(prefix="lato2-abi-smoke-") as directory:
                root = Path(directory) / "venv"
                subprocess.run([str(python), "-m", "venv", "--without-pip", str(root)], check=True, capture_output=True, text=True, timeout=60, env=deps.sanitize_subprocess_environment())
                child = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
                fingerprint = setup.interpreter_fingerprint(child)
                if fingerprint != raw:
                    raise RuntimeError("temporary venv fingerprint differs from base interpreter")
                report["venv_abi"] = deps._assert_target_python(child, normalized).payload()
                report["venv_matches"] = True
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error_code"] = getattr(exc, "code", type(exc).__name__)
        report["error"] = getattr(exc, "public_message", str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
