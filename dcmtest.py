from pydicom import dcmread
import os
import numpy as np

cnt = 0
def make3D(ds):
    arr = ds.pixel_array

def enterfile(path):
    return path + "/" + os.listdir(path)[0]

def readFile(path):
    files = os.listdir(path)
    ds_list = []
    for file in files:
        filePath = path + "/" + file
        while os.path.isdir(filePath):
            filePath = enterfile(filePath)
        ds = dcmread(filePath)
        global cnt
        #print("This is the {0} file".format(cnt),end="\r")
        cnt += 1
        ds_list.append(list(ds.pixel_array))
        if cnt == 20:
            return np.array(ds_list)

def main():
    path = "C:/Users/57035/Desktop/ELEC/3rd year/project/Xray/images in initial annotation"
    dsList = readFile(path)
    print(dsList)
if __name__ == "__main__":
    main()


