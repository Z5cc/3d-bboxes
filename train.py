import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from utils.network import Network
from utils.dataset_dl_challenge import Dataset_dl_challenge
from utils.geometry import create_bb, loss_bb
from inference import inference
from constants import TRAIN_PATH , N, EPOCHS, MODEL_PATH


def plot_losses(train_losses, test_losses):
    plt.figure()
    plt.plot(train_losses, label='train', color='black')
    plt.plot(test_losses, label='test', color='red')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(0,2)
    plt.legend()
    plt.grid()
    plt.savefig("loss.png", dpi=200)


def train():
    model = Network()
    optimizer = optim.Adam(model.parameters())
    train_data = Dataset_dl_challenge(TRAIN_PATH)
    train_loader = DataLoader(train_data, batch_size=N, shuffle=True)
    train_losses, test_losses = [], []

    for epoch in range(EPOCHS):
        epoch_loss = 0
        i = 0
        print(f'\nEPOCH: {epoch}')
        for x, bb_truth in train_loader:
            optimizer.zero_grad()

            y = model(x) # [N,3]
            bb = create_bb(y) # [N,8,3]
            loss = loss_bb(bb, bb_truth) # [N]
            loss = loss.mean() # scalar
            epoch_loss+=loss.item()
            i+=1
            print(f'train loss: {loss.item()}')
            loss.backward()
            optimizer.step()
        model_path = f'{MODEL_PATH}{epoch}'
        torch.save(model.state_dict(), model_path)
        avg_loss_test = inference(vis=False, model_path=model_path)
        avg_loss_train = epoch_loss/i
        train_losses.append(avg_loss_train)
        test_losses.append(avg_loss_test)
        plot_losses(train_losses,test_losses)

if __name__ == '__main__':
    train()
