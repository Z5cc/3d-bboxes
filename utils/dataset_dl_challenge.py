import os
import bisect
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from constants import H, W


class Dataset_dl_challenge(Dataset):
    def __init__(self, path):
        self.path = path
        self.names = []
        self.idx_cumul = []
        total = 0
        # sorting guarantees that 'names' and 'cumulative' stay consistent
        entries = sorted(os.scandir(self.path), key=lambda e: e.name)

        for i, entry in enumerate(entries):
            # create hashmap. example:  [bulk_idx: 22] -> ['911224f8-9915-11ee-9103-bbb8eae05561']
            name = entry.name
            self.names.append(name)

            # get amount of bounding boxes in 'bbox3d.npy' file for cumulative indices
            bbox3d_path = os.path.join(self.path,name,'bbox3d.npy')
            bbox3d = np.load(bbox3d_path)
            size = len(bbox3d)
            total+=size
            self.idx_cumul.append(total)


    def __len__(self):
        return self.idx_cumul[-1]


    def __getitem__(self, idx):
        # bulk_idx is idx of folder like '911224f8-9915-11ee-9103-bbb8eae05561'
        bulk_idx = bisect.bisect(self.idx_cumul, idx)
        # local_idx is for one mask or one box within that folder
        if bulk_idx==0:
            local_idx = idx
        else:
            local_idx = idx-self.idx_cumul[bulk_idx-1]

        # get paths and access '.npy' files
        name = self.names[bulk_idx]
        bulk_path = os.path.join(self.path, name)
        bbox3d_path = os.path.join(bulk_path, 'bbox3d.npy')
        mask_path = os.path.join(bulk_path, 'mask.npy')
        pc_path = os.path.join(bulk_path, 'pc.npy')

        # bb
        bb = np.load(bbox3d_path)[local_idx] # [E,8,3] -> [8,3]
        bb = torch.from_numpy(bb).float()

        # x
        mask = np.load(mask_path)[local_idx] # [E,H,W] -> [H,W]
        pc = np.load(pc_path) # [3xHxW]
        # find center of mask
        coords = np.argwhere(mask)
        h_min, w_min = coords.min(axis=0)
        h_max, w_max = coords.max(axis=0)
        ch = (h_min+h_max)//2
        cw = (w_min+w_max)//2
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
        return x, bb
    
    
    def get_names(self):
        return self.names


    def get_idx_cumul(self):
        return self.idx_cumul
    