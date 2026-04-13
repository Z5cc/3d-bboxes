from classes.dataset_dl_challenge import Dataset_dl_challenge
import pytest
from constants import TRAIN_PATH



def test_dataloader():
    dataloader = Dataset_dl_challenge(TRAIN_PATH)
    x,y = dataloader[11]
    last_idx = dataloader.idx_cumul[-1]
    assert last_idx>200

  