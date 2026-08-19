import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Patch
mpl.rcParams['axes3d.mouserotationstyle'] = 'azel'



class BB_Graphic:
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def plot(self, name, bb_inf):
        rgb_path = self.data_folder / name / 'rgb.jpg'
        bbox3d_path = self.data_folder / name / 'bbox3d.npy'
        rgb = plt.imread(rgb_path)
        bb_truth = np.load(bbox3d_path) # [E,8,3]
        self._plot(name, bb_inf, bb_truth, rgb)


    def _plot(self, name, bb_inf, bb_truth, rgb):
        fig = plt.figure(figsize=(16,10),layout='constrained')
        fig.canvas.manager.set_window_title(name)
        ax1 = plt.subplot2grid((1, 3), (0, 0), fig=fig)
        ax2 = plt.subplot2grid((1, 3), (0, 1),colspan=2,projection='3d',fig=fig)

        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')
        ax2.set_xlim(-0.25, 0.25)
        ax2.set_ylim(-0.25, 0.25)
        ax2.set_zlim(0.7, 1.4)
        self.plot_boxes(ax2,bb_inf,'red')
        self.plot_boxes(ax2,bb_truth,'blue')
        # self.plot_ground(ax2, rgb)

        ax2.legend(handles=[Patch(color='red', label='Prediction'),Patch(color='blue', label='Ground Truth')],
        loc='lower right')

        self.plot_rgb(ax1,rgb)
        ax2.view_init(vertical_axis='z',elev=-20,azim=100,roll=180) # elev=200.   only adjust azim now
        plt.show()
        plt.close(fig)


    def plot_boxes(self,ax, bbox3d, color):
        for idx,b in enumerate(bbox3d):
            self.plot_box(ax,b,idx, color)

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



    def plot_rgb(self,ax,rgb):
        ax.imshow(rgb)
        ax.axis('off')



    # def plot_ground(self, ax, rgb, z=1.4, resolution=2):
    #     rgb = rgb.astype(float) / 255.0
    #     rgb = rgb[::resolution, ::resolution]
    #     h, w = rgb.shape[:2]
    #     x = np.linspace(*ax.get_xlim(), w)
    #     y = np.linspace(*ax.get_ylim(), h)
    #     X, Y = np.meshgrid(x, y)
    #     Z = np.full_like(X, z)
    #     ax.plot_surface(X, Y, Z, facecolors=rgb, shade=False)






class Loss_Graphic:
    def __init__(self, exp_folder):
        self.exp_folder = exp_folder

    def plot_losses(self, train_loss, test_loss):
        fig, ax = plt.subplots()
        ax.plot(train_loss, label='train_loss', color='black')
        ax.plot(test_loss, label='test_loss', color='red')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_ylim(0,0.01)
        ax.legend()
        ax.grid()
        fig.savefig(self.exp_folder / "loss.png", dpi=200)
        plt.close(fig)

    def plot_RMSE(self, RMSE):
        fig, ax = plt.subplots()
        ax.plot(RMSE, label='val_RMSE', color='green')
        ax.set_xlabel('epoch')
        ax.set_ylabel('RMSE')
        ax.set_ylim(0,0.1)
        ax.legend()
        ax.grid()
        fig.savefig(self.exp_folder / "RMSE.png", dpi=200)
        plt.close(fig)
