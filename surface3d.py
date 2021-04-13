
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot(p,threshold):
    verts, faces, normals, values = measure.marching_cubes(p, threshold)
    return Poly3DCollection(verts[faces], alpha=0.70)


def main():
    p = np.load('./samples/validation/apnea/PT17A6_During_10.npy')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = plot(p, threshold=0.05)
    ax.add_collection3d(mesh)
    mesh = plot(p, threshold=0.06)
    ax.add_collection3d(mesh)
    mesh = plot(p, threshold=0.08)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()


if __name__ == '__main__':
    main()
