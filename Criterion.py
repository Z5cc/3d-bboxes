import torch
import torch.nn as nn
import torch.nn.functional as F

from Constants import PERMS



class Criterion(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, bb, bb_truth): # [N,8,3]
        # calculate delta between ground truth and all permutations of inference result
        perms = torch.tensor(PERMS, dtype=torch.long, device=bb.device) # [24,8]
        bb_perm = bb[:,perms,:] # [N,24,8,3]
        bb_truth = bb_truth.unsqueeze(1) # [N,1,8,3]
        bb_delta = bb_truth - bb_perm # [N,24,8,3]

        # we want MSE = mean_squared_L2_error    ->      squared_L2_error = x^2 + y^2 + z^2
        distances = (bb_delta ** 2).sum(dim=3) # [N,24,8]
        distances = distances.mean(dim=2) # [N,24]
        distances = distances.min(dim=1).values # [N]
        distances = distances.mean(dim=0) # scalar
        return distances
