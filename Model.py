import os
import time

import numpy as np
import torch

import torch.optim as optim
from torch.utils.data import DataLoader

from Dataset_dl_challenge import Dataset_dl_challenge
from Graphic import Graphic
from Network import Network
from Geometry import create_bb, loss_bb
from Constants import TRAIN_PATH, TEST_PATH, MODEL_PATH, EPOCHS, N, DEVICE






class Model():
    def __init__(self):
        print(f'using device: {DEVICE}')
        self.model = Network().to(DEVICE)
        self.graphic = Graphic()


    def train(self):
        optimizer = optim.Adam(self.model.parameters())
        train_data = Dataset_dl_challenge(TRAIN_PATH)
        train_loader = DataLoader(train_data, batch_size=N, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=False)
        train_losses, test_losses = [], []

        for epoch in range(EPOCHS):
            start = time.time()
            self.model.train() # in each epoch in later in inference model will be put to eval mode
            epoch_loss = 0
            print(f'\nEPOCH: {epoch}')
            for x, bb_truth in train_loader:
                x = x.to(DEVICE)
                bb_truth = bb_truth.to(DEVICE)
                optimizer.zero_grad()

                y = self.model(x) # [N,3]
                bb = create_bb(y) # [N,8,3]
                loss = loss_bb(bb, bb_truth) # [N]
                epoch_loss+=loss.item()
                print(f'train loss: {loss.item()}')
                loss.backward()
                optimizer.step()
            model_path = f'{MODEL_PATH}{epoch}'
            torch.save(self.model.state_dict(), model_path)

            avg_loss_test = self.inference(vis=False, model_path=model_path)
            avg_loss_train = epoch_loss/len(train_loader) # device by number of batches
            train_losses.append(avg_loss_train)
            test_losses.append(avg_loss_test)
            self.graphic.plot_losses(train_losses,test_losses)
            print(f'computing time for epoch {epoch}: {(time.time()-start):.2f} s')



    def inference(self, vis=True, model_path=MODEL_PATH):
        self.model.eval()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        test_data = Dataset_dl_challenge(TEST_PATH)
        test_loader = DataLoader(test_data)
        bb_all = []

        with torch.no_grad():
            total_loss=0
            for x, bb_truth in test_loader:
                x = x.to(DEVICE)
                bb_truth = bb_truth.to(DEVICE)
                y = self.model(x) # [N]
                bb = create_bb(y) # [N,8,3] with N=1
                loss = loss_bb(bb, bb_truth)
                total_loss+=loss.item()
                print(f'inference loss: {loss.item()}')
                bb = bb.cpu().numpy() # torch -> numpy
                bb_all.append(bb)
            avg_loss_test = total_loss/len(test_loader) # divide by number of batches

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