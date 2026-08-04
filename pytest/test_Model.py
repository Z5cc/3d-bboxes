import torch

from Model import Model


def test_inference(tmp_path):
    model = Model()

    model_path = tmp_path / "model.pth"
    torch.save(model.model.state_dict(), model_path)

    loss = model.inference(vis=False,model_path=model_path)
    assert isinstance(loss, float)
