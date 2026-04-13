import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils.helper import create_bb
from utils.network import Network
from utils.dataset_dl_challenge import Dataset_dl_challenge
from constants import MODEL_PATH, TEST_PATH


def plot_box(ax,b,idx):
    verts = [[b[0],b[1],b[2],b[3]],[b[4],b[5],b[6],b[7]],[b[0],b[3],b[7],b[4]],[b[3],b[2],b[6],b[7]],[b[2],b[1],b[5],b[6]],[b[0],b[1],b[5],b[4]]]
    ax.add_collection3d(Poly3DCollection(verts,facecolors='blue', linewidths=1, edgecolors='black', alpha=.1))
    for i, (x, y, z) in enumerate(b):
        ax.text(x, y, z, str(i), color='red')
    # label of box
    center = np.mean(b, axis=0)
    ax.text(
        center[0], center[1], center[2],
        str(idx),
        color='red',
        fontsize=20,
        ha='center',
        va='center'
    )

def plot_boxes(bbox3d):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    for idx,b in enumerate(bbox3d):
        plot_box(ax,b,idx)
    plt.show()








model = Network()
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

test_data = Dataset_dl_challenge(TEST_PATH)
test_loader = DataLoader(test_data)

for x,_ in test_loader:
    y = model(x)
    bb = create_bb(y)


# read bbox3d
# bbox3d = np.load('dl_challenge/911224f8-9915-11ee-9103-bbb8eae05561/bbox3d.npy')
# plot_boxes(bbox3d)
