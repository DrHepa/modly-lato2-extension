"""Dependency-light conditioning renderer used only if Open3D is unavailable.

This renderer is deliberately a fallback, not a claim of pixel equivalence to
Open3D/Filament.  It keeps the same camera controls and output contract so all
four upstream inference scripts can still run on platforms where Open3D cannot
be imported or cannot create a headless context.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw


ColorLike = Union[Sequence[float], np.ndarray]


def _to_rgb01(color: ColorLike) -> Tuple[float, float, float]:
    values = np.asarray(color, dtype=np.float64).reshape(-1)[:3]
    if values.size != 3:
        raise ValueError("color must have at least three channels")
    if values.max(initial=0.0) > 1.0 + 1e-6:
        values = values / 255.0
    values = np.clip(values, 0.0, 1.0)
    return float(values[0]), float(values[1]), float(values[2])


def _axis_index(up_axis: str) -> int:
    try:
        return {"x": 0, "y": 1, "z": 2}[up_axis.lower()]
    except KeyError as exc:
        raise ValueError("up_axis must be x, y, or z") from exc


def _orbit_eye(
    center: np.ndarray,
    distance: float,
    azimuth_deg: float,
    elevation_deg: float,
    up_axis: str,
) -> Tuple[np.ndarray, np.ndarray]:
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    horizontal = distance * np.cos(elevation)
    vertical = distance * np.sin(elevation)
    up_index = _axis_index(up_axis)
    plane_axes = [index for index in range(3) if index != up_index]
    offset = np.zeros(3, dtype=np.float64)
    offset[plane_axes[0]] = horizontal * np.cos(azimuth)
    offset[plane_axes[1]] = horizontal * np.sin(azimuth)
    offset[up_index] = vertical
    up = np.zeros(3, dtype=np.float64)
    up[up_index] = 1.0
    return center + offset, up


def _rgb8(color: ColorLike, scale: float = 1.0) -> Tuple[int, int, int]:
    rgb = np.asarray(_to_rgb01(color)) * float(scale)
    return tuple(int(value) for value in np.clip(np.rint(rgb * 255.0), 0, 255))


class SoftwareWhiteModelRenderer:
    """Small deterministic CPU rasterizer matching ``WhiteModelRenderer`` API."""

    def __init__(
        self,
        img_res: int = 512,
        mesh_color: ColorLike = (0.78, 0.78, 0.82),
        bg_color: ColorLike = (1.0, 1.0, 1.0),
        up_axis: str = "y",
        add_ground: bool = True,
        shadow: bool = True,
        elevation_range: Tuple[float, float] = (15.0, 40.0),
        azimuth_range: Tuple[float, float] = (0.0, 360.0),
        camera_distance: float = 1.8,
        fov: float = 50.0,
        ground_color: ColorLike = (0.92, 0.92, 0.92),
        sun_intensity: float = 90000.0,
        ambient_intensity: float = 32000.0,
        crop_to_object: bool = False,
        crop_padding: float = 1.2,
    ):
        del shadow, ground_color, sun_intensity, ambient_intensity
        self.img_res = int(img_res)
        if self.img_res < 8:
            raise ValueError("img_res must be at least 8")
        self.mesh_color = _to_rgb01(mesh_color)
        self.bg_color = _to_rgb01(bg_color)
        self.up_axis = up_axis.lower()
        _axis_index(self.up_axis)
        self.add_ground = bool(add_ground)
        self.elevation_range = elevation_range
        self.azimuth_range = azimuth_range
        self.camera_distance = float(camera_distance)
        self.fov = float(fov)
        self.crop_to_object = bool(crop_to_object)
        self.crop_padding = float(crop_padding)
        self._rng = np.random.default_rng()

    def _project(
        self,
        vertices: np.ndarray,
        center: np.ndarray,
        azimuth: float,
        elevation: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        eye, nominal_up = _orbit_eye(
            center,
            self.camera_distance,
            azimuth,
            elevation,
            self.up_axis,
        )
        forward = center - eye
        forward /= max(float(np.linalg.norm(forward)), 1e-12)
        right = np.cross(forward, nominal_up)
        if np.linalg.norm(right) < 1e-8:
            alternate = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, alternate)
        right /= max(float(np.linalg.norm(right)), 1e-12)
        camera_up = np.cross(right, forward)
        camera_up /= max(float(np.linalg.norm(camera_up)), 1e-12)

        relative = vertices - eye
        depth = relative @ forward
        x = relative @ right
        y = relative @ camera_up
        focal = (self.img_res * 0.5) / np.tan(np.deg2rad(self.fov) * 0.5)
        safe_depth = np.maximum(depth, 1e-6)
        projected = np.column_stack(
            (
                self.img_res * 0.5 + focal * x / safe_depth,
                self.img_res * 0.5 - focal * y / safe_depth,
            )
        )
        return projected, depth, -forward

    def _crop(self, image: Image.Image, background: Tuple[int, int, int]) -> Image.Image:
        pixels = np.asarray(image)
        mask = np.any(pixels != np.asarray(background, dtype=np.uint8), axis=2)
        ys, xs = np.where(mask)
        if not xs.size:
            return image
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        size = max(1, int(np.ceil(max(x1 - x0, y1 - y0) * self.crop_padding)))
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        left = int(np.floor(cx - size * 0.5))
        top = int(np.floor(cy - size * 0.5))
        canvas = Image.new("RGB", (size, size), background)
        src_left = max(0, left)
        src_top = max(0, top)
        src_right = min(self.img_res, left + size)
        src_bottom = min(self.img_res, top + size)
        if src_right > src_left and src_bottom > src_top:
            crop = image.crop((src_left, src_top, src_right, src_bottom))
            canvas.paste(crop, (src_left - left, src_top - top))
        return canvas.resize((self.img_res, self.img_res), Image.Resampling.LANCZOS)

    def _render_one(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        mesh_color: ColorLike,
        azimuth: float,
        elevation: float,
    ) -> np.ndarray:
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
            raise ValueError("vertices must have shape [N, 3] and be non-empty")
        if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
            raise ValueError("faces must have shape [M, 3] and be non-empty")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("face index outside the vertex array")

        center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
        projected, depth, view_direction = self._project(
            vertices, center, azimuth, elevation
        )
        background = _rgb8(self.bg_color)
        image = Image.new("RGB", (self.img_res, self.img_res), background)
        draw = ImageDraw.Draw(image)

        triangles = vertices[faces]
        normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normal_length = np.linalg.norm(normal, axis=1, keepdims=True)
        normal = normal / np.maximum(normal_length, 1e-12)
        # Two-sided diffuse lighting avoids disappearing back faces in broken
        # or inconsistently wound inputs, matching Open3D's practical behavior.
        diffuse = np.abs(normal @ view_direction)
        shade = 0.34 + 0.66 * diffuse
        face_depth = depth[faces].mean(axis=1)
        valid = np.all(depth[faces] > 1e-5, axis=1)
        # Painter's algorithm: draw far faces first.  This fallback prioritizes
        # portability; it does not claim pixel parity with Filament's z-buffer.
        order = np.argsort(face_depth)[::-1]
        for index in order:
            if not valid[index]:
                continue
            points = [tuple(point) for point in projected[faces[index]]]
            draw.polygon(points, fill=_rgb8(mesh_color, float(shade[index])))

        if self.crop_to_object:
            image = self._crop(image, background)
        return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))

    def render(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        num_views: int = 1,
        mesh_color: Optional[ColorLike] = None,
        azimuths: Optional[Sequence[float]] = None,
        elevations: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[dict]]:
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        color = self.mesh_color if mesh_color is None else _to_rgb01(mesh_color)
        images: List[np.ndarray] = []
        params: List[dict] = []
        for index in range(int(num_views)):
            azimuth = (
                float(azimuths[index])
                if azimuths is not None
                else float(rng.uniform(*self.azimuth_range))
            )
            elevation = (
                float(elevations[index])
                if elevations is not None
                else float(rng.uniform(*self.elevation_range))
            )
            images.append(
                self._render_one(vertices, faces, color, azimuth, elevation)
            )
            params.append(
                {
                    "azimuth": azimuth,
                    "elevation": elevation,
                    "distance": self.camera_distance,
                }
            )
        return images, params
