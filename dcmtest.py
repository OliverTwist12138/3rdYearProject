from pydicom import dcmread
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_scan(images):
    #plot the 2D slices
    for image in images:
        im = Image.fromarray(image)
        im = im.convert("L")
        #im.show()

def make3D(image, threshold = -400):
    #this function generates 3D image
    #unknown error with the marching_cubes function(maybe the threshold?)
    #ValueError: Surface level must be within volume data range.
    from skimage import measure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    verts, faces = measure.marching_cubes(image, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)  # 设置颜色
    ax.add_collection3d(mesh)
    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])
    ax.set_zlim(0, image.shape[2])
    plt.show()

def enterfolder(path):
    return path + "/" + os.listdir(path)[0]

def readFile(path,thickness):
    #read files and combine them to be a higher dimentional array
    files = os.listdir(path)
    ds_list = []
    cnt = 0
    for file in files:
        filePath = path + "/" + file
        while os.path.isdir(filePath):
            filePath = enterfolder(filePath)
        try:
            ds = dcmread(filePath)
        except:
            continue
        print("This is the {0} file".format(cnt),end="\r")
        cnt += 1
        ds_list.append(list(ds.pixel_array))
        if cnt == thickness:
            array = np.array(ds_list)
            return array

def main():
    path = "C:/Users/57035/Desktop/ELEC/3rd year/project/Xray/images in initial annotation"
    image = readFile(path,50)
    print(image.shape)
    #make3D(image,400)
    plot_scan(image)
if __name__ == "__main__":
    main()


