import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from utils.network import Network
from utils.dataset_dl_challenge import Dataset_dl_challenge
from utils.geometry import create_bb, loss_bb
from inference import inference
from constants import TRAIN_PATH , PERMS, N, EPOCHS, MODEL_PATH






def train():
    model = Network()
    optimizer = optim.Adam(model.parameters())
    train_data = Dataset_dl_challenge(TRAIN_PATH)
    train_loader = DataLoader(train_data, batch_size=N, shuffle=True)

    for epoch in range(EPOCHS):
        print(f'\nEPOCH: {epoch}')
        for x, bb_truth in train_loader:
            optimizer.zero_grad()

            y = model(x) # [N,3]
            bb = create_bb(y) # [N,8,3]
            loss = loss_bb(bb, bb_truth) # [N]
            loss = loss.mean() # scalar
            print(f'loss: {loss.item()}')
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), MODEL_PATH)
        inference(vis=False)


import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from utils.network import Network
from utils.dataset_dl_challenge import Dataset_dl_challenge
from utils.geometry import create_bb, loss_bb
from inference import inference
from constants import TRAIN_PATH , PERMS, N, EPOCHS, MODEL_PATH


def train():
    model = Network()
    optimizer = optim.Adam(model.parameters())
    train_data = Dataset_dl_challenge(TRAIN_PATH)
    train_loader = DataLoader(train_data, batch_size=N, shuffle=True)

    for epoch in range(EPOCHS):
        print(f'\nEPOCH: {epoch}')
        for x, bb_truth in train_loader:
            optimizer.zero_grad()

            y = model(x) # [N,3]
            bb = create_bb(y) # [N,8,3]
            loss = loss_bb(bb, bb_truth) # [N]
            loss = loss.mean() # scalar
            print(f'train loss: {loss.item()}')
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), MODEL_PATH)
        inference(vis=False)


if __name__ == '__main__':
    train()
