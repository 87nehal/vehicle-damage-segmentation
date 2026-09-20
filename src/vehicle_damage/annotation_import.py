from __future__ import annotations

import numpy as np

from .annotation_export import PASCAL_COLORS


def decode_cvat_class_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Decode indexed or RGB CVAT Segmentation Mask output to class ids."""
    value = np.asarray(mask)
    if value.ndim == 2:
        decoded = value.astype(np.uint8, copy=False)
    elif value.ndim == 3 and value.shape[2] in (3, 4):
        rgb = value[..., :3]
        decoded = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        for class_id, color in enumerate(PASCAL_COLORS[:num_classes]):
            decoded[np.all(rgb == color, axis=2)] = class_id
        if np.any(decoded == 255):
            unknown = np.unique(rgb[decoded == 255].reshape(-1, 3), axis=0)
            raise ValueError(
                f"CVAT mask contains unknown RGB colors: {unknown[:10].tolist()}"
            )
    else:
        raise ValueError("CVAT class mask must be indexed, RGB, or RGBA")
    labels = set(int(item) for item in np.unique(decoded))
    invalid = sorted(item for item in labels if item < 0 or item >= num_classes)
    if invalid:
        raise ValueError(f"CVAT mask contains invalid class ids: {invalid}")
    return decoded
