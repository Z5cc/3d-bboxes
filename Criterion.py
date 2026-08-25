import torch
import torch.nn as nn

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
        error = self.compute_error(bb_delta)
        error = error.mean(dim=2) # [N,24]
        error = error.min(dim=1).values # [N]
        error = error.mean(dim=0) # scalar
        return self.finalize(error)

    def compute_error(self, delta):
        raise NotImplementedError

    def finalize(self, error):
        return error



class MAE(Criterion):
    def compute_error(self, bb_delta):
        error = (bb_delta ** 2).sum(dim=3).sqrt() # [N,24,8]    # L2_error = root(x^2 + y^2 + z^2)                         
        return error



# we want MSE = mean_squared_L2_error -> squared_L2_error = x^2 + y^2 + z^2
class MSE(Criterion):
    def compute_error(self, bb_delta):
        error = (bb_delta ** 2).sum(dim=3) # [N,24,8]      # squared_L2_error = x^2 + y^2 + z^2
        return error



class RMSE(MSE):
    def finalize(self, error):
        return error.sqrt()
