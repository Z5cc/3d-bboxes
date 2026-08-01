import os

import numpy as np
import torch

import torch.optim as optim
from torch.utils.data import DataLoader

from Dataset_dl_challenge import Dataset_dl_challenge
from Graphic import Graphic
from Network import Network
from Geometry import create_bb, loss_bb
from Constants import TRAIN_PATH, TEST_PATH, MODEL_PATH, EPOCHS, N






class Model():
    def __init__(self):
        self.model = Network()
        self.graphic = Graphic()


    def train(self):
        optimizer = optim.Adam(self.model.parameters())
        train_data = Dataset_dl_challenge(TRAIN_PATH)
        train_loader = DataLoader(train_data, batch_size=N, shuffle=True)
        train_losses, test_losses = [], []

        for epoch in range(EPOCHS):
            epoch_loss = 0
            i = 0
            print(f'\nEPOCH: {epoch}')
            for x, bb_truth in train_loader:
                optimizer.zero_grad()

                y = self.model(x) # [N,3]
                bb = create_bb(y) # [N,8,3]
                loss = loss_bb(bb, bb_truth) # [N]
                loss = loss.mean() # scalar
                epoch_loss+=loss.item()
                i+=1
                print(f'train loss: {loss.item()}')
                loss.backward()
                optimizer.step()
            model_path = f'{MODEL_PATH}{epoch}'
            torch.save(self.model.state_dict(), model_path)
            avg_loss_test = self.inference(vis=False, model_path=model_path)
            avg_loss_train = epoch_loss/i
            train_losses.append(avg_loss_train)
            test_losses.append(avg_loss_test)
            self.graphic.plot_losses(train_losses,test_losses)



    def inference(self, vis=True, model_path=MODEL_PATH):
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        test_data = Dataset_dl_challenge(TEST_PATH)
        test_loader = DataLoader(test_data)
        bb_all = []

        # inference
        with torch.no_grad():
            total_loss=0
            i=0
            for x, bb_truth in test_loader:
                y = self.model(x) # [N]
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
                
                self.graphic.plot_all(bb_inf, bb_truth)

        return avg_loss_test