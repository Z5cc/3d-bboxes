import os
import pytest

from Dataset_dl_challenge import Dataset_dl_challenge
from Constants import TEST_PATH



def test_dataset():
    if not os.path.exists(TEST_PATH):
        pytest.skip("Test dataset is not available")
    dataset = Dataset_dl_challenge(TEST_PATH)
    # access first element
    x,y = dataset[0]
    n = len(dataset)
    x,y = dataset[n-1]
