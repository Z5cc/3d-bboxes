import os
import time
import bisect
import random

import numpy as np
import torch

from torch.utils.data import Dataset
from Constants import H, W




class Dataset_dl_challenge(Dataset):
    def __init__(self, path, prel=False, augment=True):
        self.prel = prel
        self.augment = augment
        self.path = path
        self.names = []     # ['983022a8-9915-11ee-9103-bbb8eae05561', '983022a9-9915-11ee-9103-bbb8eae05561', '983022aa-9915-11ee-9103-bbb8eae05561',...]
        self.idx_cumul = [] # [5, 13, 14, ....] # first name contains 5 object that is 5 masks
        total = 0
        # sorting guarantees that 'names' and 'cumulative' stay consistent
        entries = sorted(os.scandir(self.path), key=lambda e: e.name)

        for i, entry in enumerate(entries):
            # create hashmap. example:  [bulk_idx: 22] -> ['911224f8-9915-11ee-9103-bbb8eae05561']
            name = entry.name
            self.names.append(name)

            # get amount of bounding boxes in 'bbox3d.npy' file for cumulative indices
            bbox3d_path = self.path / name / 'bbox3d.npy'
            bbox3d = np.load(bbox3d_path)
            size = len(bbox3d)
            total+=size
            self.idx_cumul.append(total)

        if self.prel==True:
            self.bb_list = self.preload(self.load_bb)
            self.x_list = self.preload(self.load_x)

    def __len__(self):
        return self.idx_cumul[-1]

    def __getitem__(self, idx):
        if self.prel==True:
            x = self.x_list[idx]
            bb = self.bb_list[idx]
        else:
            x = self.load_x(idx)
            bb = self.load_bb(idx)
        return x, bb



    def preload(self, load_fn):
        preload_list = []
        for idx in range(len(self)):
            preload_list.append(load_fn(idx))
        return preload_list


    def load_bb(self, idx):
        bbox3d_path, mask_path, pc_path, local_idx = self.idx_to_path_and_local_idx(idx)
        bb = np.load(bbox3d_path)[local_idx] # [E,8,3] -> [8,3]
        bb = torch.from_numpy(bb).float()
        return bb


    def load_x(self, idx):
        bbox3d_path, mask_path, pc_path, local_idx = self.idx_to_path_and_local_idx(idx)
        mask = np.load(mask_path)[local_idx] # [E,H,W] -> [H,W]
        pc = np.load(pc_path) # [3xHxW]
        # find center of mask
        coords = np.argwhere(mask)
        h_min, w_min = coords.min(axis=0)
        h_max, w_max = coords.max(axis=0)
        ch = (h_min+h_max)//2
        cw = (w_min+w_max)//2
        # augment center
        if self.augment==True:
            ch, cw = self.augmentation(ch, cw)
        # move center if center is too close to border
        h = pc.shape[1]
        w = pc.shape[2]
        ch = min(max(ch,H//2-1),h-H//2-2)
        cw = min(max(cw,W//2-1),w-W//2-2)
        # concatenate mask and pc
        x = np.concatenate([mask[None,:,:],pc], axis=0)
        # then cut out H=256 and W=256 area out based on mask center
        x = x[:,ch-H//2+1:ch+H//2+1,cw-W//2+1:cw+W//2+1]
        x = torch.from_numpy(x).float()
        return x

    def augmentation(self, ch, cw, dev=10):
        ch = ch + np.random.randint(-dev, +dev+1)
        cw = cw + np.random.randint(-dev, +dev+1)
        return ch, cw

    def idx_to_path_and_local_idx(self, idx):
        # bulk_idx is idx of folder like '911224f8-9915-11ee-9103-bbb8eae05561'
        bulk_idx = bisect.bisect(self.idx_cumul, idx)
        # local_idx is for one mask or one box within that folder
        if bulk_idx==0:
            local_idx = idx
        else:
            local_idx = idx-self.idx_cumul[bulk_idx-1]

        # get paths and access '.npy' files
        name = self.names[bulk_idx]
        bulk_path = self.path / name
        bbox3d_path = bulk_path / 'bbox3d.npy'
        mask_path = bulk_path / 'mask.npy'
        pc_path = bulk_path / 'pc.npy'

        return bbox3d_path, mask_path, pc_path, local_idx
    
    def get_names(self):
        return self.names


    def get_idx_cumul(self):
        return self.idx_cumul
    