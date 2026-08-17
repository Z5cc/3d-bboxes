import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from Dataset_dl_challenge import Dataset_dl_challenge
from Graphic import Loss_Graphic
from Model import Model
from Criterion import MSE, MAE, RMSE
from val import val
from Constants import TRAIN_PATH, VAL_PATH, EPOCHS, N, DEVICE, DEFAULT_SAVE_EXP, LR



def train(exp_folder=DEFAULT_SAVE_EXP):
    print(f'using device: {DEVICE}\n')
    exp_folder.mkdir(parents=True, exist_ok=True)
    model = Model().to(DEVICE)
    criterion = MSE().to(DEVICE)
    graphic = Loss_Graphic(exp_folder)
    optimizer = optim.Adam(model.parameters(),lr=LR)
    train_data = Dataset_dl_challenge(TRAIN_PATH)
    train_loader = DataLoader(train_data, batch_size=N, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    train_loss, val_loss, val_RMSE = [], [], []


    for epoch in range(EPOCHS):
        start = time.time()
        model.train() # in each epoch in later in inference model will be put to eval mode
        total_loss = torch.zeros((), device=DEVICE)

        for x, bb_truth in train_loader:

            # GPU
            x = x.to(DEVICE, non_blocking=True)
            bb_truth = bb_truth.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()

            bb = model(x) # [N,8,3]
            loss = criterion(bb, bb_truth) # [N]
            total_loss+=loss.detach()

            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), exp_folder / 'model.pth')

        avg_val_RMSE = val(data_folder=VAL_PATH,exp_folder=exp_folder,vis=False)
        avg_val_loss = val(data_folder=VAL_PATH,exp_folder=exp_folder,criterion=criterion,vis=False)
        avg_train_loss = total_loss.item()/len(train_loader) # divide by number of batches
        val_RMSE.append(avg_val_RMSE)
        val_loss.append(avg_val_loss)
        train_loss.append(avg_train_loss)
        graphic.plot_losses(train_loss,val_loss)
        graphic.plot_RMSE(val_RMSE)
        print(f'EPOCH: {epoch}. computing time: {(time.time()-start):.2f}s. avg_train_loss: {avg_train_loss:.4f}. avg_val_loss: {avg_val_loss:.4f}.\n')



if __name__ == '__main__':
    train()
