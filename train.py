import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from Dataset_dl_challenge import Dataset_dl_challenge
from Graphic import Graphic
from Model import Model
from Criterion import Criterion
from val import val
from Constants import TRAIN_PATH, VAL_PATH, EPOCHS, N, DEVICE, DEFAULT_SAVE_EXP



def train(exp_folder=DEFAULT_SAVE_EXP):
    print(f'using device: {DEVICE}\n')
    exp_folder.mkdir(parents=True, exist_ok=True)
    model = Model().to(DEVICE)
    criterion = Criterion().to(DEVICE)
    graphic = Graphic(exp_folder = exp_folder)
    optimizer = optim.Adam(model.parameters())
    train_data = Dataset_dl_challenge(TRAIN_PATH)
    train_loader = DataLoader(train_data, batch_size=N, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    train_losses, test_losses = [], []


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

        avg_loss_test = val(data_folder=VAL_PATH,exp_folder=exp_folder,vis=False)
        avg_loss_train = total_loss.item()/len(train_loader) # divide by numtorch.save(model.state_dict(), exp_folder / 'model.pth')ber of batches
        train_losses.append(avg_loss_train)
        test_losses.append(avg_loss_test)
        graphic.plot_losses(train_losses,test_losses)
        print(f'EPOCH: {epoch}. computing time: {(time.time()-start):.2f} s\n')



if __name__ == '__main__':
    train()
