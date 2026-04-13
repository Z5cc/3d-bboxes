import torch
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from classes.network import Network
from classes.dataset_dl_challenge import Dataset_dl_challenge
from constants import TRAIN_PATH


model = Network()
optimizer = optim.Adam(model.parameters())
train_data = Dataset_dl_challenge(TRAIN_PATH)

def loss_fn(bb, bb_truth): # [8,3]
    perms = [
    # Top up
    [0,1,2,3,4,5,6,7],
    [1,2,3,0,5,6,7,4],
    [2,3,0,1,6,7,4,5],
    [3,0,1,2,7,4,5,6],
    # Bottom up
    [4,7,6,5,0,3,2,1],
    [7,6,5,4,3,2,1,0],
    [6,5,4,7,2,1,0,3],
    [5,4,7,6,1,0,3,2],
    # A up
    [4,5,1,0,7,6,2,3],
    [5,1,0,4,6,2,3,7],
    [1,0,4,5,2,3,7,6],
    [0,4,5,1,3,7,6,2],
    # B up
    [5,6,2,1,4,7,3,0],
    [6,2,1,5,7,3,0,4],
    [2,1,5,6,3,0,4,7],
    [1,5,6,2,0,4,7,3],
    # C up
    [6,7,3,2,5,4,0,1],
    [7,3,2,6,4,0,1,5],
    [3,2,6,7,0,1,5,4],
    [2,6,7,3,1,5,4,0],
    # D up
    [7,4,0,3,6,5,1,2],
    [4,0,3,7,5,1,2,6],
    [0,3,7,4,1,2,6,5],
    [3,7,4,0,2,6,5,1],
    ]

    # get index of best permutation
    perm_scores = []
    for perm in perms:
        bb_perm = bb[perm] # [8,3]
        bb_delta = bb_truth - bb_perm # [8,3]
        distances = torch.norm(bb_delta, dim=1) # [8]
        score = distances.sum() # scalar
        perm_scores.append(score)
    best_idx = perm_scores.index(min(perm_scores))

    best_perm = perms[best_idx]
    bb_perm = bb[perm]
    bb_delta = bb_truth - bb_perm
    distances = torch.norm(bb_delta, dim=1) # [8]
    distances_squared = distances ** 2
    loss = distances_squared.sum()
    return loss

def create_bb(c):
    bb = torch.tensor([[0,0,0],[0,1,0],[1,1,0],[1,0,0],[0,0,1],[0,1,1],[1,1,1],[1,0,1]]) * 0.1
    bb += c
    return bb


def train():
    for x, bb_truth in train_data:
        optimizer.zero_grad()

        y = model(x) # x:[4,H,W]
        bb = create_bb(y)

        loss = loss_fn(bb, bb_truth)
        loss.backward()
        optimizer.step()
        print(loss)
    torch.save(model.state_dict(), "model.pth")


train()
