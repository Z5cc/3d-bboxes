import torch
from Criterion import MSE
from Model import Model



def test_MSE():
    criterion = MSE()
    model = Model()
    y = torch.zeros((2, 9))

    bb = model.create_bb(y)
    loss = criterion(bb, bb)

    assert loss.shape == ()
    assert torch.allclose(loss, torch.zeros(2))
    