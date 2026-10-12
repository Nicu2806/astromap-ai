import torch

from astromap.model import PinnTransNet, physics_loss


def test_model_contract() -> None:
    model = PinnTransNet()
    logits, parameters = model(torch.zeros(2, 2, 512), torch.zeros(2, 1, 32, 512))
    assert logits.shape == (2,)
    assert parameters.shape == (2, 3)
    assert torch.all(parameters > 0)


def test_physics_loss_penalizes_impossible_duration() -> None:
    plausible = torch.tensor([[0.01, 0.10, 3.4]])
    impossible = torch.tensor([[0.01, 5.00, 3.4]])
    radius = torch.ones(1)
    mass = torch.ones(1)
    assert physics_loss(impossible, radius, mass) > physics_loss(plausible, radius, mass)
