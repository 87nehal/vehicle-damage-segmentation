import pytest
import torch

from vehicle_damage.training import _configure_trainable_parameters


class _TinySegmenter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.presence_head = torch.nn.Linear(2, 1)
        self.type_head = torch.nn.Linear(2, 5)
        self.case_classifier = torch.nn.Linear(2, 5)
        self.detail_encoder = torch.nn.Linear(2, 2)
        self.detail_refiner = torch.nn.Linear(2, 2)


def test_type_head_only_training_freezes_every_other_parameter():
    model = _TinySegmenter()

    _configure_trainable_parameters(
        model,
        train_only_case_classifier=False,
        train_only_type_head=True,
        train_only_detail_refiner=False,
        case_classifier_enabled=True,
    )

    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert trainable == {"type_head.weight", "type_head.bias"}


def test_train_only_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="exclusive"):
        _configure_trainable_parameters(
            _TinySegmenter(),
            train_only_case_classifier=True,
            train_only_type_head=True,
            train_only_detail_refiner=False,
            case_classifier_enabled=True,
        )


def test_case_head_only_training_requires_enabled_head():
    with pytest.raises(ValueError, match="requires the case classifier"):
        _configure_trainable_parameters(
            _TinySegmenter(),
            train_only_case_classifier=True,
            train_only_type_head=False,
            train_only_detail_refiner=False,
            case_classifier_enabled=False,
        )


def test_detail_only_training_freezes_parent_predictor():
    model = _TinySegmenter()

    _configure_trainable_parameters(
        model,
        train_only_case_classifier=False,
        train_only_type_head=False,
        train_only_detail_refiner=True,
        case_classifier_enabled=True,
    )

    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert trainable == {
        "detail_encoder.weight",
        "detail_encoder.bias",
        "detail_refiner.weight",
        "detail_refiner.bias",
    }
