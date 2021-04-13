
import re
import os
import scipy.io
import numpy as np
def saveCubes(mat,fileName,m):
        data = mat['data']
        measurement = data['measurement'][0,0]
        compositValue = measurement[0,0][1]
        r = len(compositValue[0][0])//20-1
        compositValue = np.array(compositValue)
        for i in range(r):
                block = compositValue[:,:,(10+20*i):(30+20*i)]
                if m == 6:
                        path = '/home/zceeyan/project/3Ddataset/samples/testing/'
                elif m == 5:
                        path = '/home/zceeyan/project/3Ddataset/samples/validation/'
                else:
                        path = '/home/zceeyan/project/3Ddataset/samples/training/'

                name = path + 'apnea/'+fileName.split('/')[-1].split('.')[0]+"_"+str(i)+'.npy'
                np.save(name,block)


def readFile():
    path = '/data/d01/CRADL-Project/MATLAB files/'
    files = os.listdir(path)
    m = 0
    for file in files:
        if re.search(r'^P', file) != None:
            filePath = path + '/' + file
            samplefiles = os.listdir(filePath)
            for samples in samplefiles:
                if re.search('_During',samples):
                    fileName = filePath + '/' + samples
                    mat = scipy.io.loadmat(fileName)
                    saveCubes(mat, fileName, m)
                    print(fileName)
            m += 1


def main():
        readFile()

if __name__ == "__main__":
    main()

