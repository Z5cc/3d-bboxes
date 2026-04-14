import torch
import torch.nn.functional as F
from constants import N, PERMS


def loss_bb(bb, bb_truth):
    # calculate delta between ground truth and all permutations of inference result
    perms = torch.tensor(PERMS, dtype=torch.long)
    bb_perm = bb[:,perms,:] # [N,24,8,3]
    bb_truth = bb_truth[:,None,:,:] # [N,1,8,3]
    bb_delta = bb_truth - bb_perm # [N,24,8,3]
    
    # from that delta vectors, calculate L2 distances, select smallest L2 distances sum from all permutations
    distances = torch.linalg.norm(bb_delta, dim=3) # [N,24,8]
    distances = distances.sum(dim=2) # [N,24]
    distances = distances.min(dim=1).values # [N]
    loss = distances ** 2
    return loss # [N]

def create_bb(y): # [N,9]
    # 0. BASE OF BOUNDING BOX
    bb = torch.tensor([[-0.5,-0.5,-0.5],[-0.5,0.5,-0.5],[0.5,0.5,-0.5],[0.5,-0.5,-0.5],
                         [-0.5,-0.5,0.5],[-0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,-0.5,0.5]],
                         dtype = torch.float)
    bb = bb[None,:,:] # [N,8,3]

    # 1. SCALE
    size = y[:,None, 3:6] # [N,1,3]
    size = F.softplus(size)
    bb = bb * size  # [N,8,3]

    # 2. ROTATE
    angles = (torch.tanh(y[:,6:9])) * (torch.pi / 4) # [N,3]
    cx, cy, cz = torch.cos(angles[:,0]), torch.cos(angles[:,1]), torch.cos(angles[:,2]) # [N]
    sx, sy, sz = torch.sin(angles[:,0]), torch.sin(angles[:,1]), torch.sin(angles[:,2]) # [N]
    R = torch.zeros((y.shape[0], 3, 3)) # [N,3,3]

    R[:,0,0] = cy * cz
    R[:,0,1] = cz * sx * sy - cx * sz
    R[:,0,2] = cx * cz * sy + sx * sz

    R[:,1,0] = cy * sz
    R[:,1,1] = cx * cz + sx * sy * sz
    R[:,1,2] = -cz * sx + cx * sy * sz

    R[:,2,0] = -sy
    R[:,2,1] = cy * sx
    R[:,2,2] = cx * cy

    bb = torch.matmul(bb, R.transpose(1,2)) # [N,8,3]=[N,8,3]*[N,3,3]

    # 3. SHIFT
    center = y[:,None,0:3] # [N,1,3]
    bb = bb + center

    return bb # [N,8,3]
