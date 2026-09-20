import pytest
import torch

from vehicle_damage.checkpoint_soup import average_state_dicts


def test_average_state_dicts_interpolates_float_and_preserves_integer():
    first = {
        "weight": torch.tensor([0.0, 2.0]),
        "counter": torch.tensor(3, dtype=torch.int64),
    }
    second = {
        "weight": torch.tensor([2.0, 6.0]),
        "counter": torch.tensor(3, dtype=torch.int64),
    }

    result = average_state_dicts(first, second, 0.25)

    torch.testing.assert_close(result["weight"], torch.tensor([0.5, 3.0]))
    assert result["counter"].item() == 3


def test_average_state_dicts_rejects_incompatible_states():
    with pytest.raises(ValueError, match="keys differ"):
        average_state_dicts({"a": torch.ones(1)}, {"b": torch.ones(1)}, 0.5)
    with pytest.raises(ValueError, match="non-floating tensor differs"):
        average_state_dicts(
            {"counter": torch.tensor(1)},
            {"counter": torch.tensor(2)},
            0.5,
        )
