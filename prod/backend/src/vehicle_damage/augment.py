from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def _resize_all(images: list[Image.Image], size: tuple[int, int]) -> list[Image.Image]:
    result = [TF.resize(images[0], size, InterpolationMode.BILINEAR, antialias=True)]
    result.extend(TF.resize(x, size, InterpolationMode.NEAREST) for x in images[1:])
    return result


def _resize_short_side(images: list[Image.Image], short_side: int) -> list[Image.Image]:
    width, height = images[0].size
    scale = short_side / min(width, height)
    new_height = max(1, round(height * scale))
    new_width = max(1, round(width * scale))
    return _resize_all(images, (new_height, new_width))


def _letterbox(images: list[Image.Image], size: int) -> list[Image.Image]:
    width, height = images[0].size
    scale = size / max(width, height)
    resized = _resize_all(images, (max(1, round(height * scale)), max(1, round(width * scale))))
    width, height = resized[0].size
    pad_w, pad_h = size - width, size - height
    left, top = pad_w // 2, pad_h // 2
    return [TF.pad(x, [left, top, pad_w - left, pad_h - top], fill=0) for x in resized]


def _synthetic_glare(image: Image.Image, rng: random.Random) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = image.size
    cx, cy = rng.randint(0, w), rng.randint(0, h)
    rx = rng.randint(max(8, w // 12), max(9, w // 3))
    ry = rng.randint(max(8, h // 20), max(9, h // 4))
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=(255, 255, 245, rng.randint(35, 105)))
    overlay = overlay.filter(ImageFilter.GaussianBlur(max(3, min(rx, ry) // 3)))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _synthetic_dirt(image: Image.Image, rng: random.Random) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = image.size
    for _ in range(rng.randint(8, 35)):
        radius = rng.randint(2, max(3, min(w, h) // 40))
        x, y = rng.randint(0, w), rng.randint(0, h)
        color = rng.choice(((80, 65, 45), (115, 100, 75), (65, 65, 65)))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, rng.randint(15, 70)))
    overlay = overlay.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.6)))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _synthetic_shadow(image: Image.Image, rng: random.Random) -> Image.Image:
    """Overlay a soft cast shadow without changing the supervision masks."""
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = image.size
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    angle = rng.uniform(0, math.pi)
    length = 2.0 * math.hypot(w, h)
    half_width = rng.uniform(0.06, 0.24) * min(w, h)
    dx, dy = math.cos(angle) * length, math.sin(angle) * length
    px, py = -math.sin(angle) * half_width, math.cos(angle) * half_width
    polygon = [
        (cx - dx + px, cy - dy + py),
        (cx + dx + px, cy + dy + py),
        (cx + dx - px, cy + dy - py),
        (cx - dx - px, cy - dy - py),
    ]
    draw.polygon(polygon, fill=(12, 18, 25, rng.randint(35, 115)))
    overlay = overlay.filter(ImageFilter.GaussianBlur(rng.uniform(4.0, 18.0)))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _sensor_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    """Approximate high-ISO phone noise with a deterministic per-call seed."""
    array = np.asarray(image, dtype=np.float32)
    generator = np.random.default_rng(rng.randrange(2**32))
    sigma = rng.uniform(2.0, 12.0)
    noisy = array + generator.normal(0.0, sigma, size=array.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), mode="RGB")


def _degrade_resolution(image: Image.Image, rng: random.Random) -> Image.Image:
    """Remove fine detail without moving semantic labels."""
    factor = rng.uniform(0.20, 0.55)
    low_size = (
        max(1, round(image.width * factor)),
        max(1, round(image.height * factor)),
    )
    return image.resize(low_size, Image.Resampling.BILINEAR).resize(
        image.size, Image.Resampling.BILINEAR
    )


def _random_viewpoint(
    images: list[Image.Image], rng: random.Random
) -> list[Image.Image]:
    """Apply one label-aligned affine or mild projective camera perturbation."""
    width, height = images[0].size
    if rng.random() < 0.5:
        angle = rng.uniform(-12.0, 12.0)
        translate = (
            round(rng.uniform(-0.05, 0.05) * width),
            round(rng.uniform(-0.05, 0.05) * height),
        )
        scale = rng.uniform(0.92, 1.08)
        shear = [rng.uniform(-6.0, 6.0), rng.uniform(-3.0, 3.0)]
        return [
            TF.affine(
                value,
                angle,
                translate,
                scale,
                shear,
                interpolation=(
                    InterpolationMode.BILINEAR if index == 0 else InterpolationMode.NEAREST
                ),
                fill=0,
            )
            for index, value in enumerate(images)
        ]

    max_x = max(1, round(width * 0.09))
    max_y = max(1, round(height * 0.09))
    startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    endpoints = [
        [rng.randint(0, max_x), rng.randint(0, max_y)],
        [width - 1 - rng.randint(0, max_x), rng.randint(0, max_y)],
        [width - 1 - rng.randint(0, max_x), height - 1 - rng.randint(0, max_y)],
        [rng.randint(0, max_x), height - 1 - rng.randint(0, max_y)],
    ]
    return [
        TF.perspective(
            value,
            startpoints,
            endpoints,
            interpolation=(
                InterpolationMode.BILINEAR if index == 0 else InterpolationMode.NEAREST
            ),
            fill=0,
        )
        for index, value in enumerate(images)
    ]


@dataclass
class RobustAugment:
    size: int
    training: bool = True
    positive_crop_probability: float = 0.85
    profile: str = "baseline_v1"

    def __post_init__(self) -> None:
        if self.profile not in {"baseline_v1", "robust_v2", "robust_v3"}:
            raise ValueError(f"unknown augmentation profile: {self.profile}")

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
        exterior: Image.Image,
        hard_negative: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = random
        items = [image.convert("RGB"), mask, exterior, hard_negative]
        if self.training:
            source_mask = np.asarray(items[1])
            source_classes = np.unique(source_mask[source_mask > 0])
            selected_class = None
            if source_classes.size and rng.random() < self.positive_crop_probability:
                selected_class = int(source_classes[rng.randrange(source_classes.size)])
            original_items = items
            full_scene = self.profile == "robust_v3" and rng.random() < 0.30
            if full_scene:
                candidate = _letterbox(items, self.size)
                candidate_mask = np.asarray(candidate[1])
                retains_damage = not source_classes.size or (candidate_mask > 0).any()
                retains_selected = selected_class is None or (
                    candidate_mask == selected_class
                ).any()
                if retains_damage and retains_selected:
                    items = candidate
                else:
                    full_scene = False
            if not full_scene:
                scale = rng.uniform(0.55, 1.65)
                target = max(96, round(self.size * scale))
                items = _resize_short_side(items, target)
                if selected_class is not None and not (
                    np.asarray(items[1]) == selected_class
                ).any():
                    # Very thin labels can disappear during a full-car downscale.
                    # Retry as a close-up before selecting the crop.
                    items = _resize_short_side(
                        original_items, max(target, self.size * 4)
                    )
                width, height = items[0].size
                pad_w, pad_h = max(0, self.size - width), max(0, self.size - height)
                pad_left = rng.randint(0, pad_w) if pad_w else 0
                pad_top = rng.randint(0, pad_h) if pad_h else 0
                items = [
                    TF.pad(
                        value,
                        [pad_left, pad_top, pad_w - pad_left, pad_h - pad_top],
                        fill=0,
                    )
                    for value in items
                ]
                width, height = items[0].size
                mask_array = np.asarray(items[1])
                if selected_class is not None:
                    positive_y, positive_x = np.nonzero(mask_array == selected_class)
                    choice = rng.randrange(positive_x.size)
                    point_x, point_y = int(positive_x[choice]), int(positive_y[choice])
                    left_min = max(0, point_x - self.size + 1)
                    left_max = min(point_x, width - self.size)
                    top_min = max(0, point_y - self.size + 1)
                    top_max = min(point_y, height - self.size)
                    left = rng.randint(left_min, left_max)
                    top = rng.randint(top_min, top_max)
                else:
                    top = rng.randint(0, height - self.size)
                    left = rng.randint(0, width - self.size)
                items = [TF.crop(x, top, left, self.size, self.size) for x in items]
            if rng.random() < 0.5:
                items = [TF.hflip(x) for x in items]
            if self.profile in {"robust_v2", "robust_v3"} and rng.random() < 0.45:
                original_geometry = items
                transformed = _random_viewpoint(items, rng)
                transformed_mask = np.asarray(transformed[1])
                retains_damage = not source_classes.size or (transformed_mask > 0).any()
                retains_selected = selected_class is None or (
                    transformed_mask == selected_class
                ).any()
                if retains_damage and retains_selected:
                    items = transformed
                else:
                    # Do not let viewpoint augmentation erase the rare class
                    # that the positive crop deliberately retained.
                    items = original_geometry
            image = items[0]
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.55, 1.5))
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.55, 1.55))
            image = ImageEnhance.Color(image).enhance(rng.uniform(0.55, 1.45))
            if rng.random() < 0.3:
                image = _synthetic_glare(image, rng)
            if rng.random() < 0.25:
                image = _synthetic_dirt(image, rng)
            if self.profile in {"robust_v2", "robust_v3"} and rng.random() < 0.25:
                image = _synthetic_shadow(image, rng)
            if self.profile in {"robust_v2", "robust_v3"} and rng.random() < 0.2:
                image = _sensor_noise(image, rng)
            if self.profile == "robust_v3" and rng.random() < 0.35:
                image = _degrade_resolution(image, rng)
            if rng.random() < 0.2:
                image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.6)))
            if rng.random() < 0.25:
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=rng.randint(35, 85))
                buffer.seek(0)
                image = Image.open(buffer).convert("RGB")
            items[0] = image
        else:
            items = _letterbox(items, self.size)

        image_tensor = TF.to_tensor(items[0])
        image_tensor = TF.normalize(image_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        mask_tensor = torch.from_numpy(np.array(items[1], dtype=np.int64, copy=True))
        exterior_tensor = torch.from_numpy(np.array(items[2], dtype=np.uint8, copy=True) > 0).float()
        hard_tensor = torch.from_numpy(np.array(items[3], dtype=np.uint8, copy=True) > 0).float()
        return image_tensor, mask_tensor, exterior_tensor, hard_tensor
