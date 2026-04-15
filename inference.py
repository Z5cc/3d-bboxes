import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils.geometry import create_bb
from utils.network import Network
from utils.dataset_dl_challenge import Dataset_dl_challenge
from utils.graphic import Graphic
from utils.geometry import loss_bb
from constants import MODEL_PATH, TEST_PATH


def inference(vis=True, model_path=MODEL_PATH):
    model = Network()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    test_data = Dataset_dl_challenge(TEST_PATH)
    test_loader = DataLoader(test_data)
    bb_all = []

    # inference
    with torch.no_grad():
        total_loss=0
        i=0
        for x, bb_truth in test_loader:
            y = model(x) # [N]
            bb = create_bb(y) # [N,8,3] with N=1
            loss = loss_bb(bb, bb_truth)
            loss = loss.mean()
            total_loss+=loss.item()
            i+=1
            print(f'inference loss: {loss.item()}')
            bb = bb.numpy() # torch -> numpy
            bb_all.append(bb)
        avg_loss_test = total_loss/i

    # group bb with idx_cumul
    idx_cumul = test_data.get_idx_cumul()
    idx_cumul_zero = [0]+idx_cumul
    bb_per_folder = [
        bb_all[start:end]
        for start, end in zip(idx_cumul_zero[:-1], idx_cumul_zero[1:])
    ]
    bb_per_folder = [np.concatenate(one_folder, axis=0) for one_folder in bb_per_folder]

    if vis==True:
        # visualization
        for name, bb_inf in zip(test_data.get_names(), bb_per_folder):
            bbox3d_path = os.path.join(TEST_PATH,name,'bbox3d.npy')
            bb_truth = np.load(bbox3d_path) # [E,8,3]
            graphic = Graphic()
            graphic.plot_all(bb_inf, bb_truth)

    return avg_loss_test


if __name__ == '__main__':
    inference(model_path='model.pth')
