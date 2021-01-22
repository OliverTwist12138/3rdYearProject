import numpy as np
import os
path = '/home/zceeyan/project/3Ddataset/'
def normalization(images,folderPath, i, filename):
    max = np.amax(images)
    min = np.amin(images)
    images_normalized = (images-min)/(max-min)
    filename = path + 'normalized/'+ folderPath + '/' + i + filename
    np.save(filename,images_normalized)

def readFile():
    folders = os.listdir(path)
    for folderPath in folders:
        filepath = path + folderPath + '/'  # get folders in 3d dataset: test/train/validation
        if os.path.isdir(filepath):
            for i in ['apnea/','nonapnea/']:
                Nfilepath = filepath + i
                files = os.listdir(Nfilepath)
                for filename in files:
                    file = Nfilepath + filename
                    images = np.load(file)
                    normalization(images, folderPath, i, filename)

def main():
        readFile()

if __name__ == "__main__":
    main()
