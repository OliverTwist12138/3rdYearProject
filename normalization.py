import numpy as np
import os
def normalization(images,file):
    max = np.amax(images)
    min = np.amin(images)
    images_normalized = (images-min)/(max-min)
    filename = 'normalized_'+file
    np.save(filename,images_normalized)
def readFile():
    path = '/home/zceeyan/project/3Ddataset/'
    filepath = os.listdir(path)
    for files in filepath:
        for file in files:
            images = np.load(file)
            normalization(images,file)

def main():
        readFile()

if __name__ == "__main__":
    main()
