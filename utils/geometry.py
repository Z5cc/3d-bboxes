import torch

def create_bb(y): # [N,6]
    # create base of bb, that is a cube
    base = torch.tensor([[-0.5,-0.5,-0.5],[-0.5,0.5,-0.5],[0.5,0.5,-0.5],[0.5,-0.5,-0.5],
                         [-0.5,-0.5,0.5],[-0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,-0.5,0.5]],
                         dtype = torch.float)
    # bb = base * width + center
    bb = base[None,:,:] * y[:,None,3:6] + y[:,None,0:3] # [N,8,3]
    return bb
