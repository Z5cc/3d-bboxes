import torch

from Model import Model
from Dataset_dl_challenge import Dataset_dl_challenge
from val import val
from Constants import TEST_PATH, H, W



def test_inference(tmp_path):
    dataset = Dataset_dl_challenge(TEST_PATH)
    # access first element
    x,y = dataset[0]
    model = Model()

    x = x.unsqueeze(0) # [C,H,W] -> [N,C,H,W] 
    output = model(x)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1,8,3)

def test_network():
    model = Model()

    x = torch.zeros((1, 4, H, W))
    output = model(x)

    assert output.shape == (1,8,3)

def test_create_bb():
    model = Model()
    y = torch.zeros((2, 9))
    bb = model.create_bb(y)
    assert bb.shape == (2, 8, 3)
