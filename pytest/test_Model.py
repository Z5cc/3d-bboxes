import os

import pytest
import torch

from Model import Model
from Constants import TEST_PATH


def test_inference(tmp_path):
    if not os.path.exists(TEST_PATH):
        pytest.skip("Test dataset is not available")

    model = Model()

    model_path = tmp_path / "model.pth"
    torch.save(model.model.state_dict(), model_path)

    loss = model.inference(vis=False,model_path=model_path)
    assert isinstance(loss, float)
