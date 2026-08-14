import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np



class Graphic:
    def __init__(self, exp_folder):
        self.exp_folder = exp_folder

    def plot_box(self, ax, b, idx, color):
        verts = [[b[0],b[1],b[2],b[3]],[b[4],b[5],b[6],b[7]],[b[0],b[3],b[7],b[4]],[b[3],b[2],b[6],b[7]],[b[2],b[1],b[5],b[6]],[b[0],b[1],b[5],b[4]]]
        ax.add_collection3d(Poly3DCollection(verts,facecolors=color, linewidths=1, edgecolors='black', alpha=.1))
        for i, (x, y, z) in enumerate(b):
            ax.text(x, y, z, str(i), color=color)
        # label of box
        center = np.mean(b, axis=0)
        ax.text(
            center[0], center[1], center[2],
            str(idx),
            color=color,
            fontsize=20,
            ha='center',
            va='center')

    def plot_boxes(self,ax, bbox3d, color):
        for idx,b in enumerate(bbox3d):
            self.plot_box(ax,b,idx, color)

    def plot_all(self, bb_inf, bb_truth):
        fig = plt.figure(figsize=(16,10))
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        self.plot_boxes(ax,bb_inf,'red')
        self.plot_boxes(ax,bb_truth,'blue')
        plt.show()
        plt.close(fig)
        

    def plot_losses(self, train_losses, test_losses):
        fig, ax = plt.subplots()
        ax.plot(train_losses, label='train', color='black')
        ax.plot(test_losses, label='test', color='red')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_ylim(0,2)
        ax.legend()
        ax.grid()
        fig.savefig(self.exp_folder / "loss.png", dpi=200)
        plt.close(fig)
        