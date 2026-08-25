import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from BB_Dataset import BB_Dataset
from Graphic import Loss_Graphic
from Model import Model
from Criterion import MSE, MAE, RMSE
from val import evaluate
from Constants import TRAIN_PATH, VAL_PATH, EPOCHS, N, DEVICE, DEFAULT_SAVE_EXP, LR, NUM_WORKERS, DEFAULT_EXP_NAME, AUG



def train(exp_folder):
    print(f'using device {DEVICE} for experiment {DEFAULT_EXP_NAME}\n')
    exp_folder.mkdir(parents=True, exist_ok=True)
    model = Model().to(DEVICE)
    criterion = MSE().to(DEVICE)
    graphic = Loss_Graphic(exp_folder)
    optimizer = optim.Adam(model.parameters(),lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=20,factor=0.7)
    train_data = BB_Dataset(TRAIN_PATH,aug=AUG)
    val_data = BB_Dataset(VAL_PATH,aug=False)
    train_loader = DataLoader(train_data, batch_size=N, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_data, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)
    train_loss_epochs, val_loss_epochs, val_RMSE_epochs = [], [], []

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

        val_loss = evaluate(val_loader,model,criterion)
        train_loss = total_loss.item()/len(train_loader) # divide by number of batches
        val_loss_epochs.append(val_loss)
        train_loss_epochs.append(train_loss)
        graphic.plot_losses(train_loss_epochs,val_loss_epochs)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f'EPOCH: {epoch}. computing time: {(time.time()-start):.2f}s. train_loss: {train_loss:.6f}. val_loss: {val_loss:.6f}. lr: {current_lr:.6f}.')
        scheduler.step(val_loss)



if __name__ == '__main__':
    train(DEFAULT_SAVE_EXP)
