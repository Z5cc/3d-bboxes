import os

import numpy as np
import torch

from torch.utils.data import DataLoader

from Dataset_dl_challenge import Dataset_dl_challenge
from Graphic import BB_Graphic
from Model import Model
from Criterion import RMSE
from Constants import VAL_PATH, DEFAULT_INFERENCE_EXP, DEVICE, NUM_WORKERS



def val(data_folder, exp_folder, criterion=RMSE().to(DEVICE), vis=True):
    graphic = BB_Graphic(data_folder)
    model = Model().to(DEVICE)
    criterion = criterion
    model.eval()
    model.load_state_dict(torch.load(exp_folder / "model.pth", map_location=DEVICE, weights_only=True))
    test_data = Dataset_dl_challenge(data_folder)
    test_loader = DataLoader(test_data, num_workers=NUM_WORKERS)
    bb_all = []

    with torch.inference_mode():
        total_loss = torch.zeros((), device=DEVICE)
        for x, bb_truth in test_loader:

            # GPU
            x = x.to(DEVICE)
            bb_truth = bb_truth.to(DEVICE)

            bb = model(x) # [N,8,3] with N=1
            loss = criterion(bb, bb_truth) # [N]
            total_loss+=loss.detach()

            bb = bb.cpu().numpy() # torch -> numpy
            bb_all.append(bb)

        avg_val_loss = total_loss.item()/len(test_loader) # divide by number of batches
        # print(f'avg_val_loss: {avg_val_loss:.4f}.')

    # group bb with idx_cumul
    idx_cumul = test_data.get_idx_cumul()
    bb_per_folder = [
        bb_all[start:end]
        for start, end in zip([0] + idx_cumul[:-1], idx_cumul)
    ]
    bb_per_folder = [np.concatenate(one_folder, axis=0) for one_folder in bb_per_folder]

    if vis==True:
        # visualization
        for name, bb_inf in zip(test_data.get_names(), bb_per_folder):
            graphic.plot_all(name, bb_inf)

    return avg_val_loss



if __name__ == '__main__':
    val(data_folder=VAL_PATH,exp_folder=DEFAULT_INFERENCE_EXP)
