import numpy as np
import os
path = '/home/zceeyan/project/3Ddataset/samples'


def normalization(images,folderpath, i, filename):
    max = np.amax(images)
    min = np.amin(images)
    images_normalized = (images-min)/(max-min)
    filename = '/home/zceeyan/project/3Ddataset/normalized/'+ folderpath + '/' + i + filename
    print(filename)
    np.save(filename,images_normalized)


def main():
    folders = os.listdir(path)
    for folderPath in folders:
        filepath = path + '/' + folderPath # get folders in 3d dataset: test/train/validation
        print(filepath)
        if os.path.isdir(filepath):
            for i in ['/apnea/','/nonapnea/']:
                Nfilepath = filepath + i
                files = os.listdir(Nfilepath)
                for filename in files:
                    file = Nfilepath + filename
                    images = np.load(file)
                    normalization(images, folderPath, i, filename)


if __name__ == "__main__":
    main()
