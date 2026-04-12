import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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


def plot_mask(ax,mask,pc,idx):
    pc = np.transpose(pc, (1, 2, 0))
    pc_m = np.where(mask[..., None], pc, np.nan)
    valid_points = pc_m[~np.isnan(pc_m).any(axis=2)]
    center = np.mean(valid_points, axis=0)
    ax.text(
        center[0], center[1], center[2],
        str(idx),
        color='green',
        fontsize=20,
        ha='center',
        va='center'
    )


def plot_boxes(bbox3d,masks,pc):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    for idx,b in enumerate(bbox3d):
        plot_box(ax,b,idx)
    for idx,mask in enumerate(masks):
        plot_mask(ax,mask,pc,idx)
    plt.show()






# read bbox3d
bbox3d = np.load('dl_challenge/911224f8-9915-11ee-9103-bbb8eae05561/bbox3d.npy')
# read pc
pc = np.load('dl_challenge/911224f8-9915-11ee-9103-bbb8eae05561/pc.npy')
# read masks
masks = np.load('dl_challenge/911224f8-9915-11ee-9103-bbb8eae05561/mask.npy')


plot_boxes(bbox3d,masks,pc)
