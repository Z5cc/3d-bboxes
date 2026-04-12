import torch
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from network import Network
from dataloader import Dataloader


model = Network()
optimizer = optim.Adam(model.parameters())
dataloader = Dataloader()


def train():
    for X,y in dataloader:

    y = model(x)
    loss = criterion(y,label)
    loss.backwards()
    optimizer.step()


train()
