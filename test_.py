from dataloader import Dataloader
import pytest



def test_dataloader():
    dataloader = Dataloader()
    x,y = dataloader[11]
    last_idx = dataloader.idx_cumul[-1]
    assert last_idx>200
  