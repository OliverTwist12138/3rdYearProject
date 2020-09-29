from pydicom import dcmread
import os

cnt = 0
def readFile(path):
    files = os.listdir(path)
    dsCube = []
    for file in files:
        filePath = path + "/" + file
        if os.path.isdir(filePath):
            readFile(filePath)
        else:
            ds = dcmread(filePath)
            global cnt
            print("This is the {0} file".format(cnt),end="\r")
            cnt += 1
            dsCube.append(ds)
        if len(dsCube) == 20:
            break
    return dsCube
def main():
    path = "C:/Users/57035/Desktop/ELEC/3rd year/RSNA Chest Xray/images in initial annotation"
    dsList = readFile(path)
    print(dsList)
if __name__ == "__main__":
    main()


