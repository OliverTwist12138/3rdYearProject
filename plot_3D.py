import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_3d(image, threshold=-1):
    # Position the scan upright, so the head of the patient would be at
    # the top facing the camera
        p = image.transpose(2,1,0)
        verts, faces, norm, val = measure.marching_cubes(p, threshold, allow_degenerate=True)
#       verts, faces = measure.marching_cubes(p, threshold)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of
    # triangles
        mesh = Poly3DCollection(verts[faces], alpha=0.70)
        face_color = [0.45, 0.45, 0.75]
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        ax.set_xlim(0, p.shape[0])
        ax.set_ylim(0, p.shape[1])
        ax.set_zlim(0, p.shape[2])
        plt.show()
def sample_stack(stack, rows=4, cols=4):
        fig,ax = plt.subplots(rows,cols,figsize=[12,12])
        print('begin')
        for i in range(rows*cols):
                ax[int(i/rows),int(i % rows)].set_title('slice %d' % i)
                ax[int(i/rows),int(i % rows)].imshow(stack[:,:,i],cmap='gray')
                ax[int(i/rows),int(i % rows)].axis('off')
        print('show')
        plt.show()

def plot_ct_scan(scan):
        f, plots = plt.subplots(int(scan.shape[2] / 20) + 1, 4, figsize=(50, 50))
        for i in range(0, scan.shape[2], 5):
                        plots[int(i / 20), int((i % 20) / 5)].axis('off')
                        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[:,:,i], cmap=plt.cm.bone)
                plt.show()
def main():
        '''
        path='./validation/apnea'
        files = os.listdir(path)
        for file in files:
                filePath = path + "/" + file
                image=np.load(filePath)
#       image=np.load('./validation/apnea/PT31A3_During_30.npy')
                for i in range(image.shape[2]):
                        print(np.amin(image[:,:,i]))
        '''
        image = np.load('./validation/apnea/PT17A4_During_4.npy')
        sample_stack(image)
#       plot_ct_scan(image)
#       print(image.shape[2])
#       plot_3d(image)
if __name__ == '__main__':
        main()

