import pytest
import torch

from vehicle_damage.metrics import binary_auroc


def test_binary_auroc_handles_ordering_and_ties():
    assert binary_auroc(
        torch.tensor([0.9, 0.8, 0.2, 0.1]),
        torch.tensor([1, 1, 0, 0]),
    ) == 1.0
    assert binary_auroc(
        torch.tensor([0.5, 0.5]), torch.tensor([1, 0])
    ) == 0.5
    assert binary_auroc(torch.tensor([0.5]), torch.tensor([1])) is None


def test_binary_auroc_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        binary_auroc(torch.tensor([0.1, 0.2]), torch.tensor([1]))
