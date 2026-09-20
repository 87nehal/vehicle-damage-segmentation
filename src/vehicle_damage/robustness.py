from __future__ import annotations

import hashlib
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .augment import (
    _random_viewpoint,
    _sensor_noise,
    _synthetic_dirt,
    _synthetic_glare,
    _synthetic_shadow,
)


STRESS_VERSION = "synthetic_stress_v1"

STRESS_CONDITIONS = (
    "clean",
    "underexposure",
    "overexposure",
    "glare",
    "dirt",
    "shadow",
    "sensor_noise",
    "blur",
    "low_resolution",
    "viewpoint",
    "distance",
)

STRESS_DESCRIPTIONS = {
    "clean": "Unmodified RGB image and indexed target.",
    "underexposure": "Global brightness multiplied by 0.35.",
    "overexposure": "RGB image blended 38% toward white.",
    "glare": "Seeded soft elliptical specular overlay from robust_v2 augmentation.",
    "dirt": "Seeded translucent dirt spots from robust_v2 augmentation.",
    "shadow": "Seeded soft cast-shadow overlay from robust_v2 augmentation.",
    "sensor_noise": "Seeded Gaussian sensor noise from robust_v2 augmentation.",
    "blur": "Gaussian blur with radius 2.2 pixels.",
    "low_resolution": "Five-fold bilinear downsample followed by upsample.",
    "viewpoint": "Seeded label-aligned affine or projective transform.",
    "distance": "Image and target scaled to 58% and centered on a border-median canvas.",
}


def stress_seed(sample_identity: str, condition: str) -> int:
    """Return a stable seed independent of Python's randomized string hash."""
    digest = hashlib.sha256(f"stress-v1:{sample_identity}:{condition}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _distance_view(image: Image.Image, target: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Shrink the observed vehicle while retaining the original output canvas."""
    width, height = image.size
    new_size = (max(1, round(width * 0.58)), max(1, round(height * 0.58)))
    reduced_image = image.resize(new_size, Image.Resampling.LANCZOS)
    reduced_target = target.resize(new_size, Image.Resampling.NEAREST)
    pixels = np.asarray(image.convert("RGB"))
    border = np.concatenate(
        (pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]), axis=0
    )
    fill = tuple(int(value) for value in np.median(border, axis=0))
    image_canvas = Image.new("RGB", image.size, fill)
    target_canvas = Image.new("L", target.size, 0)
    offset = ((width - new_size[0]) // 2, (height - new_size[1]) // 2)
    image_canvas.paste(reduced_image, offset)
    target_canvas.paste(reduced_target, offset)
    return image_canvas, target_canvas


def apply_stress(
    image: Image.Image,
    target: Image.Image,
    condition: str,
    *,
    seed: int,
) -> tuple[Image.Image, Image.Image]:
    """Apply one deterministic stress condition with label-safe geometry."""
    if condition not in STRESS_CONDITIONS:
        raise ValueError(f"unknown stress condition: {condition!r}")
    rgb = image.convert("RGB")
    mask = target.convert("L")
    if rgb.size != mask.size:
        raise ValueError("image and target must have the same size")
    rng = random.Random(seed)
    if condition == "clean":
        return rgb.copy(), mask.copy()
    if condition == "underexposure":
        return ImageEnhance.Brightness(rgb).enhance(0.35), mask.copy()
    if condition == "overexposure":
        return Image.blend(rgb, Image.new("RGB", rgb.size, "white"), 0.38), mask.copy()
    if condition == "glare":
        return _synthetic_glare(rgb, rng), mask.copy()
    if condition == "dirt":
        return _synthetic_dirt(rgb, rng), mask.copy()
    if condition == "shadow":
        return _synthetic_shadow(rgb, rng), mask.copy()
    if condition == "sensor_noise":
        return _sensor_noise(rgb, rng), mask.copy()
    if condition == "blur":
        return rgb.filter(ImageFilter.GaussianBlur(radius=2.2)), mask.copy()
    if condition == "low_resolution":
        low_size = (max(1, rgb.width // 5), max(1, rgb.height // 5))
        degraded = rgb.resize(low_size, Image.Resampling.BILINEAR).resize(
            rgb.size, Image.Resampling.BILINEAR
        )
        return degraded, mask.copy()
    if condition == "viewpoint":
        transformed = _random_viewpoint([rgb, mask], rng)
        return transformed[0].convert("RGB"), transformed[1].convert("L")
    if condition == "distance":
        return _distance_view(rgb, mask)
    raise AssertionError("unreachable")
