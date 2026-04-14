import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


class Graphic:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')


    def plot_box(self, b, idx, color):
        verts = [[b[0],b[1],b[2],b[3]],[b[4],b[5],b[6],b[7]],[b[0],b[3],b[7],b[4]],[b[3],b[2],b[6],b[7]],[b[2],b[1],b[5],b[6]],[b[0],b[1],b[5],b[4]]]
        self.ax.add_collection3d(Poly3DCollection(verts,facecolors=color, linewidths=1, edgecolors='black', alpha=.1))
        for i, (x, y, z) in enumerate(b):
            self.ax.text(x, y, z, str(i), color=color)
        # label of box
        center = np.mean(b, axis=0)
        self.ax.text(
            center[0], center[1], center[2],
            str(idx),
            color=color,
            fontsize=20,
            ha='center',
            va='center'
    )

    def plot_boxes(self, bbox3d, color):
        for idx,b in enumerate(bbox3d):
            self.plot_box(b,idx, color)


    def plot_all(self, bb_inf, bb_truth):
        self.plot_boxes(bb_inf,'red')
        self.plot_boxes(bb_truth,'blue')
        plt.show()
