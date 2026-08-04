import torch

from Geometry import create_bb, loss_bb



def test_loss_bb():
    y = torch.zeros((2, 9))

    bb = create_bb(y)
    loss = loss_bb(bb, bb)

    assert loss.shape == (2,)
    assert torch.allclose(loss, torch.zeros(2))


def test_create_bb():
    y = torch.zeros((2, 9))
    bb = create_bb(y)
    assert bb.shape == (2, 8, 3)
