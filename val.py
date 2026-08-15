import os

import numpy as np
import torch

from torch.utils.data import DataLoader

from Dataset_dl_challenge import Dataset_dl_challenge
from Graphic import Graphic
from Model import Model
from Criterion import Criterion
from Constants import VAL_PATH, DEFAULT_INFERENCE_EXP, DEVICE



def val(data_folder, exp_folder, vis=True):
    print(f'using device: {DEVICE}\n')
    graphic = Graphic(exp_folder)
    model = Model().to(DEVICE)
    criterion = Criterion().to(DEVICE)
    model.eval()
    model.load_state_dict(torch.load(exp_folder / "model.pth", map_location=DEVICE, weights_only=True))
    test_data = Dataset_dl_challenge(data_folder)
    test_loader = DataLoader(test_data, num_workers=4, persistent_workers=True)
    bb_all = []

    with torch.inference_mode():
        total_loss = torch.zeros((), device=DEVICE)
        for x, bb_truth in test_loader:

            # GPU
            x = x.to(DEVICE, non_blocking=True)
            bb_truth = bb_truth.to(DEVICE, non_blocking=True)

            bb = model(x) # [N,8,3] with N=1
            loss = criterion(bb, bb_truth) # [N]
            total_loss+=loss.detach()

            bb = bb.cpu().numpy() # torch -> numpy
            bb_all.append(bb)

        avg_loss_test = total_loss.item()/len(test_loader) # divide by number of batches

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
            bbox3d_path = os.path.join(data_folder,name,'bbox3d.npy')
            bb_truth = np.load(bbox3d_path) # [E,8,3]
            
            graphic.plot_all(bb_inf, bb_truth)

    return avg_loss_test



if __name__ == '__main__':
    val(data_folder=VAL_PATH,exp_folder=DEFAULT_INFERENCE_EXP)
