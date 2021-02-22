import scipy.io
import numpy as np
mat = scipy.io.loadmat('/data/d01/CRADL-Project/MATLAB files/Patient 4/PT4A1_Duing.mat')
data = mat['data']
dataType = data.dtype
print(dataType)
patient = data['patient'][0,0]
events = data['events'][0,0]
measurement = data['measurement'][0,0]
imageRate = data['imageRate'][0,0]
events = data['events'][0,0]
ver = data['ver'][0,0]
injctionPattern = data['injctionPattern'][0,0]
TICVersion = data['TICVersion'][0,0]
SBCVersion = data['SBCVersion'][0,0]
SensorBelt = data['SensorBelt'][0,0]
compositValue = measurement[0,0][1]
print(measurement)
'''
'''
for i in range(len(compositValue)):
        print(compositValue[i])
        length = len(compositValue[i])
        print(length)
        print(compositValue[i][0].size)
print("####",len(measurement[0,0][1]))
'''
r = len(compositValue[0][0])//20+1
#print(range)
print(compositValue[:][:].shape)
print('###########')
compositValue = np.array(compositValue)
print(compositValue[:,:,0:20].shape)

for i in range(r):
        block = compositValue[:,:,(20*i):(20+20*i)]
        print('block',i,block)

'''





