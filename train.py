import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from utils.network import Network
from utils.dataset_dl_challenge import Dataset_dl_challenge
from utils.helper import create_bb
from constants import TRAIN_PATH , PERMS, N, MODEL_PATH




def loss_bb(bb, bb_truth):
    # calculate delta between ground truth and all permutations of inference result
    perms = torch.tensor(PERMS, dtype=torch.long)
    bb_perm = bb[:,perms,:] # [N,24,8,3]
    bb_truth = bb_truth[:,None,:,:] # [N,1,8,3]
    bb_delta = bb_truth - bb_perm # [N,24,8,3]
    
    # from that delta vectors, calculate L2 distance, select smallest sum from all permutations
    distances = torch.linalg.norm(bb_delta, dim=3) # [N,24,8]
    distances = distances.sum(dim=2) # [N,24]
    distances = distances.min(dim=1).values # [N]
    loss = distances ** 2
    return loss # [N]


def train():
    model = Network()
    optimizer = optim.Adam(model.parameters())
    train_data = Dataset_dl_challenge(TRAIN_PATH)
    train_loader = DataLoader(train_data, batch_size=N, shuffle=True)

    for x, bb_truth in train_loader:
        optimizer.zero_grad()

        y = model(x) # [N,3]
        bb = create_bb(y) # [N,8,3]
        loss = loss_bb(bb, bb_truth) # [N]
        loss = loss.mean() # scalar
        loss.backward()
        optimizer.step()
        print(loss)
    torch.save(model.state_dict(), MODEL_PATH)




train()
