from unittest.mock import patch
import numpy as np

from Graphic import BB_Graphic, Loss_Graphic
from Constants import DEFAULT_SAVE_EXP, TEST_PATH



def test_plot_losses(tmp_path): # tmp_path and monkeypatch are 'built-in fixtures' from pytest
    graphic = Loss_Graphic(tmp_path)
    graphic.plot_losses([1.0, 0.5], [1.2, 0.7])
    assert (tmp_path / 'loss.png').exists()


def test_plot(visualize=False):
    graphic = BB_Graphic(TEST_PATH)
    box1 = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ])
    box2 = np.array([
        [2, 2, 0],
        [3, 2, 0],
        [3, 3, 0],
        [2, 3, 0],
        [2, 2, 2],
        [3, 2, 2],
        [3, 3, 2],
        [2, 3, 2],
    ])
    rgb = np.zeros((400, 400, 3))


    if visualize:
        graphic._plot('test',[box1,box2], [box1,box2],rgb)
    else:
        with patch('matplotlib.pyplot.show') as mock_show:
            graphic._plot('test', [box1,box2], [box1,box2],rgb)
    assert mock_show.called
