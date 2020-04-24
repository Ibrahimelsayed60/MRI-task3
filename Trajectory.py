import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import time
from mpl_toolkits import mplot3d
from matplotlib.patches import FancyArrowPatch
from itertools import count
from mpl_toolkits.mplot3d import axes3d
import librosa
import timer
import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn
import cv2

colors = {  'bg':       [1,1,1],
            'circle':   [0,0,0,.03],
            'axis':     [.5,.5,.5],
            'text':     [.05,.05,.05],
            'spoilText':[.5,0,0]}
ax = 0
ay =0
j = 0
# variable used to get the next index of the array
index = count()
minimum=10
maximum=25
delta =0.1

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = mplot3d.proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)





def transform_kspace_to_image(k, dim=None, img_shape=None):
    """ Computes the Fourier transform from k-space to image space
    along a given or all dimensions
    :param k: k-space data
    :param dim: vector of dimensions to transform
    :param img_shape: desired shape of output image
    :returns: data in image space (along transformed dimensions)
    """
    if not dim:
        dim = range(k.ndim)

    img = fftshift(ifftn(ifftshift(k, axes=dim), s=img_shape, axes=dim), axes=dim)
    img *= np.sqrt(np.prod(np.take(img.shape, dim)))
    return img


def transform_image_to_kspace(img, dim=None, k_shape=None):
    """ Computes the Fourier transform from image space to k-space space
    along a given or all dimensions
    :param img: image space data
    :param dim: vector of dimensions to transform
    :param k_shape: desired shape of output k-space data
    :returns: data in k-space (along transformed dimensions)
    """
    if not dim:
        dim = range(img.ndim)

    k = fftshift(fftn(ifftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
    k /= np.sqrt(np.prod(np.take(img.shape, dim)))
    return k


def read_img(filename):
    img = cv2.imread(filename, 0)
    return img


def plot():
    global ax
    ax.set_axis_off()
    ax.plot([-5, 5], [0, 0], [0, 0], c=colors['axis'], zorder=-1)  # x-axis
    ax.text(3.08, 0, 0, r'$x^\prime$', horizontalalignment='center', color=colors['text'])
    ax.plot([0, 0], [-5, 5], [0, 0], c=colors['axis'], zorder=-1)  # y-axis
    ax.text(0, 3.12, 0, r'$y^\prime$', horizontalalignment='center', color=colors['text'])
    ax.plot([0, 0], [0, 0], [-5, 5], c=colors['axis'], zorder=-1)  # z-axis
    ax.text(0, 0, 3.05, r'$z$', horizontalalignment='center', color=colors['text'])


# rotation function it takes flipAngle and magnetizaton vector magnitude
def rotate(flipAngle: 'float', Mo: 'float') -> np.ndarray:
    flipAngle = (flipAngle * 3.14) / 180
    rotation_matrix = np.array([[np.cos(flipAngle), -1 * np.sin(flipAngle)], [np.sin(flipAngle), np.cos(flipAngle)]])
    magnetization_vector = np.transpose(np.array([Mo, 0]))
    return np.matmul(rotation_matrix, magnetization_vector)

def relaxition(i, array:'np.ndarray', Mo:'float', T1: 'float', T2_star: 'float', W1, W2 ):
    global j,ax, a
    exp_x_spin1 = array[1] * np.exp(-1 * np.arange(0, 10 * T1, 0.5) / T2_star)* np.cos(W1*np.arange(0, 10 * T1, 0.5))
    exp_y_spin1 = array[1] * np.exp(-1 * np.arange(0, 10 * T1, 0.5) / T2_star) * np.sin(W1 * np.arange(0, 10 * T1, 0.5))
    exp_z_spin1 = (np.ones(len(exp_x_spin1)) * Mo) - ((Mo - array[0]) * np.exp(-1 * np.arange(0, 10 * T1, 0.5) / T1))
    exp_x_spin2 = array[1] * np.exp(-1 * np.arange(0, 10 * T1, 0.5) / T2_star) * np.cos(W2*np.arange(0, 10 * T1, 0.5))
    exp_y_spin2 = array[1] * np.exp(-1 * np.arange(0, 10 * T1, 0.5) / T2_star) * np.sin(W2*np.arange(0, 10 * T1, 0.5))
    exp_z_spin2 = (np.ones(len(exp_x_spin2)) * Mo) - ((Mo - array[0]) * np.exp(-1 * np.arange(0, 10 * T1, 0.5) / T1))
    if j < len(exp_z_spin1):
        # DRAW 3D GRAPH
        ax.cla()
        plot()
        a = Arrow3D([0, exp_x_spin1[j]], [0, exp_y_spin1[j]], [0, exp_z_spin1[j]], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
        b = Arrow3D([0, exp_x_spin2[j]], [0, exp_y_spin2[j]], [Mo, exp_z_spin2[j]+Mo], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
        ax.add_artist(a)
        ax.add_artist(b)
        # INCREMENT THE COUNTER FOR ANOTHER CALL FOR THE FUNCTON
        j = j + 1



def main():
    global ax
    fig = plt.figure()
    ax = fig.gca(projection='3d', xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),fc=colors['bg'])
    plot()
    # ROTATE THE VECTOR
    array1 = rotate(90, 10)
    relax = FuncAnimation(plt.gcf(), relaxition, fargs=(array1, 5, 4, 3,100, 200,), interval=1000)
    plt.show()

if __name__ == "__main__":
    main()