from __future__ import annotations

from collections.abc import Mapping

import torch


def average_state_dicts(
    state_a: Mapping[str, torch.Tensor],
    state_b: Mapping[str, torch.Tensor],
    weight_b: float,
) -> dict[str, torch.Tensor]:
    """Average compatible fine-tuned weights while failing closed on mismatch."""
    if not 0 <= weight_b <= 1:
        raise ValueError("weight_b must be in [0, 1]")
    if state_a.keys() != state_b.keys():
        missing_a = sorted(state_b.keys() - state_a.keys())
        missing_b = sorted(state_a.keys() - state_b.keys())
        raise ValueError(
            f"checkpoint state keys differ; missing from A={missing_a}, missing from B={missing_b}"
        )
    result: dict[str, torch.Tensor] = {}
    for key in state_a:
        value_a = state_a[key]
        value_b = state_b[key]
        if value_a.shape != value_b.shape or value_a.dtype != value_b.dtype:
            raise ValueError(f"incompatible tensor for {key}")
        if value_a.is_floating_point() or value_a.is_complex():
            result[key] = torch.lerp(value_a, value_b, weight_b)
        else:
            if not torch.equal(value_a, value_b):
                raise ValueError(f"non-floating tensor differs for {key}")
            result[key] = value_a.clone()
    return result
