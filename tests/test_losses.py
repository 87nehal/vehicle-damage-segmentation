import torch

from vehicle_damage.losses import RecallOrientedLoss


def test_loss_is_finite_and_backpropagates():
    presence = torch.randn(2, 1, 12, 12, requires_grad=True)
    damage_type = torch.randn(2, 2, 12, 12, requires_grad=True)
    exterior = torch.randn(2, 1, 12, 12, requires_grad=True)
    target = torch.zeros(2, 12, 12, dtype=torch.long)
    target[:, 3:6, 3:6] = 1
    hard_negative = torch.zeros_like(target, dtype=torch.float32)
    hard_negative[:, 7:9, 7:9] = 1
    criterion = RecallOrientedLoss(3)
    loss, parts = criterion(
        {"presence": presence, "type": damage_type, "exterior": exterior}, target,
        torch.ones_like(target, dtype=torch.float32), hard_negative,
    )
    assert torch.isfinite(loss)
    assert set(parts) == {
        "total", "presence_focal", "type", "classification", "tversky", "exterior"
    }
    loss.backward()
    assert presence.grad is not None
    assert damage_type.grad is not None


def test_exterior_only_batch_does_not_train_damage_heads():
    presence = torch.randn(2, 1, 8, 8, requires_grad=True)
    damage_type = torch.randn(2, 2, 8, 8, requires_grad=True)
    exterior_logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = torch.zeros(2, 8, 8, dtype=torch.long)
    criterion = RecallOrientedLoss(3)
    loss, parts = criterion(
        {"presence": presence, "type": damage_type, "exterior": exterior_logits},
        target,
        torch.ones_like(target, dtype=torch.float32),
        torch.zeros_like(target, dtype=torch.float32),
        damage_supervised=torch.zeros(2),
        exterior_supervised=torch.ones(2),
    )
    loss.backward()
    assert parts["presence_focal"] == 0.0
    assert parts["type"] == 0.0
    assert parts["tversky"] == 0.0
    assert exterior_logits.grad is not None and exterior_logits.grad.abs().sum() > 0
    assert presence.grad is not None and presence.grad.abs().sum() == 0
    assert damage_type.grad is not None and damage_type.grad.abs().sum() == 0


def test_multilabel_case_loss_backpropagates_to_case_head():
    target = torch.zeros(2, 8, 8, dtype=torch.long)
    target[0, 1:3, 1:3] = 1
    target[1, 4:7, 4:7] = 2
    outputs = {
        "presence": torch.randn(2, 1, 8, 8, requires_grad=True),
        "type": torch.randn(2, 2, 8, 8, requires_grad=True),
        "exterior": torch.randn(2, 1, 8, 8, requires_grad=True),
        "case_logits": torch.randn(2, 2, requires_grad=True),
    }
    criterion = RecallOrientedLoss(
        3,
        classification_weight=2.0,
        classification_pos_weights=[1.0, 2.0],
    )
    loss, parts = criterion(
        outputs,
        target,
        torch.ones_like(target, dtype=torch.float32),
        torch.zeros_like(target, dtype=torch.float32),
    )
    loss.backward()
    assert parts["classification"] > 0
    assert outputs["case_logits"].grad is not None
    assert outputs["case_logits"].grad.abs().sum() > 0
