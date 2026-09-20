from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .augment import RobustAugment
from .manifest import Sample


class DamageDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        image_size: int,
        training: bool,
        augmentation_profile: str = "baseline_v1",
    ) -> None:
        self.samples = samples
        self.transform = RobustAugment(
            image_size,
            training=training,
            profile=augmentation_profile,
        )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _optional_mask(path: Path | None, size: tuple[int, int], *, default: int) -> Image.Image:
        if path is None:
            return Image.new("L", size, color=default)
        with Image.open(path) as source:
            mask = source.convert("L")
        if mask.size != size:
            raise ValueError(f"optional mask size mismatch: {path}")
        return mask

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        with Image.open(sample.image) as source:
            image = source.convert("RGB")
        with Image.open(sample.mask) as source:
            mask = source.convert("L")
        if mask.size != image.size:
            raise ValueError(f"image/mask size mismatch: {sample.image} / {sample.mask}")
        # A missing exterior mask is unknown, not an all-vehicle label.  The
        # supervision flag below keeps it out of the exterior-head loss.
        exterior = self._optional_mask(sample.exterior_mask, image.size, default=0)
        hard_negative = self._optional_mask(sample.hard_negative_mask, image.size, default=0)
        image_t, mask_t, exterior_t, hard_t = self.transform(image, mask, exterior, hard_negative)
        return {
            "image": image_t,
            "mask": mask_t,
            "exterior": exterior_t,
            "hard_negative": hard_t,
            "damage_supervised": sample.damage_supervised,
            "exterior_supervised": sample.exterior_mask is not None,
            "path": str(sample.image),
            "group_id": sample.group_id,
        }
