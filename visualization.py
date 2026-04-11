import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_box(ax,b):
    verts = [[b[0],b[1],b[2],b[3]],[b[4],b[5],b[6],b[7]],[b[0],b[3],b[7],b[4]],[b[3],b[2],b[6],b[7]],[b[2],b[1],b[5],b[6]],[b[0],b[1],b[5],b[4]]]
    ax.add_collection3d(Poly3DCollection(verts,facecolors='blue', linewidths=1, edgecolors='black', alpha=.1))
    for i, (x, y, z) in enumerate(b):
        ax.text(x, y, z, str(i), color='red')

def plot_boxes(bbox3d):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    for b in bbox3d:
        plot_box(ax,b)
    plt.show()




# read bbox3d
bbox3d = np.load('dl_challenge_example/bbox3d.npy')

plot_boxes(bbox3d)
