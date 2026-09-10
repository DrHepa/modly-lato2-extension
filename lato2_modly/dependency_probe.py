"""Shared dependency import probe and bounded, credential-free failure details.

Constraints limit versions; they do not make conditional dependencies mandatory.
Markers below are audited against huggingface_hub v0.36.0 setup.py. Evaluate them
in the target interpreter, preserving platform.machine() casing (Windows AMD64).
"""
from __future__ import annotations

import os
import re
import subprocess
from typing import Mapping

CONDITIONAL_DISTRIBUTION_MARKERS = {
    "hf-xet": (
        "platform_machine == 'x86_64' or platform_machine == 'amd64' or "
        "platform_machine == 'arm64' or platform_machine == 'aarch64'"
    ),
}

METADATA_PROBE = r'''
import importlib
import importlib.metadata
import json
import os
import sys

# Flush progress before imports/native calls, including calls that can terminate
# the interpreter without a Python traceback (e.g. a missing DLL or driver).
def _stage(name):
    print("LATO2_PROBE_STAGE=" + name, file=sys.stderr, flush=True)

_stage("import:packaging.markers")
from packaging.markers import Marker
expected = json.loads(os.environ["MODLY_LATO2_EXPECTED_DISTS"])
conditional = json.loads(os.environ["MODLY_LATO2_CONDITIONAL_DISTS"])
conditional_absent = []
for name, wanted in expected.items():
    _stage("version:" + name)
    marker = conditional.get(name)
    required = marker is None or Marker(marker).evaluate()
    try:
        actual = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        if required:
            raise
        conditional_absent.append(name)
        continue
    # Inactive markers allow absence, never a silently incompatible installation.
    if actual != wanted:
        raise RuntimeError(f"{name} version {actual!r} != {wanted!r}")
'''

IMPORT_PROBE = METADATA_PROBE + r'''
_stage("import:numpy")
import numpy
_stage("import:trimesh")
import trimesh
_stage("import:tqdm")
import tqdm
_stage("import:PIL")
import PIL
_stage("import:ninja")
import ninja
_stage("import:psutil")
import psutil
_stage("import:cv2")
import cv2
_stage("import:huggingface_hub")
import huggingface_hub
_stage("import:plyfile")
import plyfile
_stage("import:zstandard")
import zstandard
_stage("import:easydict")
import easydict
_stage("import:einops")
import einops
_stage("import:filelock")
import filelock
_stage("import:torch")
import torch
_stage("import:torchvision")
import torchvision

if os.environ.get("MODLY_LATO2_OPEN3D_SMOKE") == "1":
    _stage("import:open3d")
    import open3d
    if os.environ.get("MODLY_LATO2_OPEN3D_RENDER_SMOKE") == "1":
        _stage("open3d:renderer")
        renderer = open3d.visualization.rendering.OffscreenRenderer(16, 16)
        rendered = numpy.asarray(renderer.render_to_image())
        if rendered.ndim != 3 or rendered.shape[:2] != (16, 16):
            raise RuntimeError("Open3D OffscreenRenderer returned an invalid image")
        del renderer
'''


def failure_detail(exc: BaseException, *, paths: Mapping[str, str] | None = None) -> str:
    """Return bounded stderr/stdout evidence, never the command's inline script."""
    def text(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value if isinstance(value, str) else ""

    stderr = text(getattr(exc, "stderr", ""))
    stdout = text(getattr(exc, "stdout", "")) or text(getattr(exc, "output", ""))
    stages = re.findall(r"(?m)^LATO2_PROBE_STAGE=([A-Za-z0-9_.:-]{1,100})\r?$", stderr)
    stage = stages[-1] if stages else "unreported"
    if isinstance(exc, subprocess.TimeoutExpired):
        status = "timeout"
    elif isinstance(exc, subprocess.CalledProcessError):
        status = f"exit={exc.returncode}"
    else:
        status = type(exc).__name__
    # Preserve the exception line at the tail without echoing the whole command,
    # script, environment, or an unbounded third-party log into Modly's UI.
    evidence = re.sub(r"(?m)^LATO2_PROBE_STAGE=.*\r?\n?", "", stderr)
    evidence = evidence[-16000:] or stdout[-16000:]
    if not evidence.strip():
        evidence = "No child diagnostic output; the process may have failed to start or terminated natively."
        if isinstance(exc, OSError):
            evidence += f" errno={exc.errno}; winerror={getattr(exc, 'winerror', None)}."
    replacements = dict(paths or {})
    for name in ("USERPROFILE", "HOME"):
        if os.environ.get(name):
            replacements[os.environ[name]] = "<user-home>"
    for source, label in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        if source:
            evidence = evidence.replace(source, label).replace(source.replace("\\", "/"), label)
    evidence = re.sub(r"(?i)\b(?:https?|ftp|socks[45]?)://[^\s/@]+@", "<redacted-url-userinfo>@", evidence)
    evidence = re.sub(r"(?i)\b(?:authorization|proxy-authorization|cookie|set-cookie)\s*:[^\r\n]+", "<redacted-header>", evidence)
    evidence = re.sub(r"(?i)\b(token|api[_-]?key|password|passwd|secret|signature|sig)\s*[:=]\s*(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;&]+)", r"\1=<redacted>", evidence)
    evidence = re.sub(r"(?i)\bhf_[A-Za-z0-9._-]{10,}\b|\bbearer\s+[A-Za-z0-9._~+/=-]{8,}", "<redacted-credential>", evidence)
    evidence = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", evidence)
    evidence = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", evidence)
    return f"stage={stage}; {status}\n{evidence.strip()[-6000:]}"
