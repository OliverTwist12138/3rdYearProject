from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from model import get_model
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

model = get_model(width=32, height=32, depth=20)
x_test = np.load('x_test.npy')
x_test = x_test[0]
image_arr = np.expand_dims(x_test, axis=3)
image_arr = np.expand_dims(image_arr,axis=0)
model.load_weights("./model/3d_image_classification.h5")
layer_1 = K.function([model.layers[0].input], [
    model.layers[7].output])
print(model.layers[0].input.shape)

print(image_arr.shape)
f1 = layer_1([image_arr])[0]
print(f1[0].shape)
f1=f1[0]

p = f1[:,:,:,8]
p = p[:,:,0]
print(p.shape)

for _ in range(32):#512表示特征图通道数
    p = f1[:, :, :, :, _]
    plt.subplot(8, 4, _ + 1)#将特征图显示为32行16列
    plt.subplot(8, 4, _ + 1)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    verts, faces, normals, values = measure.marching_cubes(p, 0.05)
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()
    plt.axis('off')
    plt.show()



