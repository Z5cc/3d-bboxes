import pytest

from Dataset_dl_challenge import Dataset_dl_challenge
from Constants import TEST_PATH



def test_dataset():
    dataloader = Dataset_dl_challenge(TEST_PATH)
    # access first element
    x,y = dataloader[0]
    assert 0.0806 < y[0][0] < 0.0807
    # access last element
    x,y = dataloader[49]
    assert 0.233 < y[0][0] < 0.234
