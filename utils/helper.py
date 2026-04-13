import torch

def create_bb(c):
    base = torch.tensor([[0,0,0],[0,1,0],[1,1,0],[1,0,0],[0,0,1],[0,1,1],[1,1,1],[1,0,1]], dtype = torch.float)
    base *= 0.1 # [8,3]
    base = base[None,:,:] # [1,8,3]
    c = c[:,None,:] # [N,3] -> [N,1,3]
    bb = base+c # broadcasting to [N,8,3]
    return bb
