import numpy as np
import torch

from torch.utils.data import DataLoader

from Dataset_dl_challenge import Dataset_dl_challenge
from Graphic import BB_Graphic
from Model import Model
from Criterion import RMSE
from Constants import VAL_PATH, DEFAULT_INFERENCE_EXP, DEVICE, NUM_WORKERS



def evaluate(data_loader, model, criterion, return_bb=False): # only part used by train.py
    model.eval()

    total_loss = torch.zeros((), device=DEVICE)
    bb_all = []

    with torch.inference_mode():
        for x, bb_truth in data_loader:
            # GPU
            x = x.to(DEVICE, non_blocking=True)
            bb_truth = bb_truth.to(DEVICE, non_blocking=True)

            bb = model(x) # [N,8,3] with N=1
            loss = criterion(bb, bb_truth) # [N]
            total_loss += loss.detach()

            if return_bb:
                bb = bb.cpu().numpy() # torch -> numpy
                bb_all.append(bb)

    val_loss = total_loss.item()/len(data_loader) # divide by number of batches
    if return_bb:
        bb_all = np.concatenate(bb_all, axis=0)
        return val_loss, bb_all
    return val_loss



def val(exp_folder, data_folder): # in case i do test.py, i load this function to test.py script and input other TEST PATH
    val_data = Dataset_dl_challenge(data_folder)
    data_loader = DataLoader(val_data, num_workers=NUM_WORKERS)
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(exp_folder / "model.pth", map_location=DEVICE, weights_only=True))
    criterion = RMSE().to(DEVICE)

    val_loss, bb_all=evaluate(data_loader, model, criterion, return_bb=True)

    graphic = BB_Graphic(data_folder)
    graphic.plot(bb_all, val_data.get_idx_cumul(), val_data.get_names())



if __name__ == '__main__':
    val(DEFAULT_INFERENCE_EXP, VAL_PATH)
