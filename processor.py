"""Dependency-free Modly PROCESS bootstrap for LATO.2.

The host protocol is kept on a duplicated descriptor. File descriptors 1 and
2 are redirected before importing the runtime so banners from Python packages,
CUDA libraries, or native extensions can never corrupt Modly's NDJSON stream.
"""

from __future__ import annotations

import json
import os
import sys
from typing import TextIO


# Declared literally so Modly's static validator can audit the real boundary.
MODLY_PROCESS_CONTRACT = (
    "stdin",
    "json",
    "progress",
    "log",
    "done",
    "error",
    "result",
    "filePath",
    "workspaceDir",
    "tempDir",
    "nodeId",
)

BOOTSTRAP_ERROR = (
    "[PROCESS_BOOTSTRAP_FAILED] LATO.2 initialization failed. "
    "Run Repair for this extension and try again."
)


def _emit_bootstrap_error(channel: TextIO) -> None:
    channel.write(
        json.dumps(
            {"type": "error", "message": BOOTSTRAP_ERROR},
            ensure_ascii=True,
        )
        + "\n"
    )
    channel.flush()


def _main() -> int:
    protocol_fd: int | None = None
    channel: TextIO | None = None
    try:
        protocol_fd = os.dup(1)
        discard_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(discard_fd, 1)
            os.dup2(discard_fd, 2)
        finally:
            os.close(discard_fd)

        channel = os.fdopen(
            protocol_fd,
            "w",
            encoding="utf-8",
            errors="strict",
            buffering=1,
            newline="\n",
        )
        protocol_fd = None

        # This import must remain after both public descriptors are isolated.
        from lato2_modly.runtime import run_protocol

        return run_protocol(sys.stdin, channel)
    except BaseException:
        if channel is not None:
            try:
                _emit_bootstrap_error(channel)
            except BaseException:
                pass
        elif protocol_fd is not None:
            try:
                line = (
                    json.dumps(
                        {"type": "error", "message": BOOTSTRAP_ERROR},
                        ensure_ascii=True,
                    )
                    + "\n"
                ).encode("utf-8")
                os.write(protocol_fd, line)
            except BaseException:
                pass
        return 1
    finally:
        if channel is not None:
            try:
                channel.close()
            except BaseException:
                pass
        elif protocol_fd is not None:
            try:
                os.close(protocol_fd)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(_main())
