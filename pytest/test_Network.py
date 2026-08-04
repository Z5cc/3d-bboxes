import torch

from Network import Network
from Constants import H, W


def test_network():
    network = Network()

    x = torch.zeros((1, 4, H, W))
    y = network(x)

    assert y.shape == (1, 9)
