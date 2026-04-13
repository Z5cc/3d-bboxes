from classes.dataset_dl_challenge import Dataset_dl_challenge
import pytest
from constants import TEST_PATH


# for this test to work the following folders need to be in the dl_challenge_test:
def test_dataset():
    dataloader = Dataset_dl_challenge(TEST_PATH)
    # access first element
    x,y = dataloader[0]
    assert 0.0806 < y[0][0] < 0.0807
    # access last element
    x,y = dataloader[49]
    assert 0.233 < y[0][0] < 0.234
