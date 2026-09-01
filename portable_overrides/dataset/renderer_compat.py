"""Open3D-first renderer adapter with an explicit portable fallback."""

from __future__ import annotations

import os
import warnings

from dataset.software_renderer import SoftwareWhiteModelRenderer


RENDERER_ENV = "LATO2_RENDERER"


class WhiteModelRenderer:
    """Use the upstream Open3D renderer unless it cannot run.

    ``LATO2_RENDERER=auto`` is the default.  ``open3d`` makes an Open3D failure
    fatal, while ``software`` is an explicit diagnostic/compatibility override.
    """

    def __init__(self, *args, **kwargs):
        mode = os.environ.get(RENDERER_ENV, "auto").strip().lower()
        if mode not in {"auto", "open3d", "software"}:
            raise ValueError(
                f"invalid {RENDERER_ENV}={mode!r}; expected auto, open3d, or software"
            )
        self._args = args
        self._kwargs = kwargs
        self._mode = mode
        self._using_fallback = mode == "software"
        if self._using_fallback:
            self._renderer = SoftwareWhiteModelRenderer(*args, **kwargs)
            return
        try:
            from dataset.mesh_render import WhiteModelRenderer as Open3DRenderer

            self._renderer = Open3DRenderer(*args, **kwargs)
        except Exception as exc:
            if mode == "open3d":
                raise
            self._activate_fallback(exc)

    def _activate_fallback(self, reason: BaseException) -> None:
        warnings.warn(
            "Open3D conditioning renderer is unavailable; using Modly's "
            f"portable software fallback ({type(reason).__name__}: {reason}). "
            "The fallback keeps the camera/output contract but is not pixel-equivalent.",
            RuntimeWarning,
            stacklevel=2,
        )
        self._renderer = SoftwareWhiteModelRenderer(*self._args, **self._kwargs)
        self._using_fallback = True

    def render(self, *args, **kwargs):
        try:
            return self._renderer.render(*args, **kwargs)
        except Exception as exc:
            if self._mode != "auto" or self._using_fallback:
                raise
            self._activate_fallback(exc)
            return self._renderer.render(*args, **kwargs)
