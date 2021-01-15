
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img3d = np.load('./validation/apnea/PT17A6_During_10.npy')
img_shape = [32,32,20]
# plot 3 orthogonal slices
a1 = plt.subplot(2, 2, 1)
plt.imshow(img3d[:, :, img_shape[2]//2])
#a1.set_aspect(ax_aspect)

a2 = plt.subplot(2, 2, 2)
plt.imshow(img3d[:, img_shape[1]//2, :])
#a2.set_aspect(sag_aspect)

a3 = plt.subplot(2, 2, 3)
plt.imshow(img3d[img_shape[0]//2, :, :].T)
#a3.set_aspect(cor_aspect)
plt.show()

'''
print(np.amin(image))
plt.imshow(image[:,:,5])
plt.show()
'''
#for i in range(32):
#       print(image[i,:,5])
Z=image
size=Z.shape
Y=np.arange(0,size[0],1)
X=np.arange(0,size[1],1)

X,Y=np.meshgrid(X,Y)
fig=plt.figure()
ax=fig.gca(projection='3d')
ax.plot_surface(X,Y,Z)
plt.show()
'''
fig = plt.figure(figsize=(40,40))
ax = fig.gca(projection='3d')
ax.view_init(30,angle)
ax.voxels(x,y,x,shade=False)
plt.show()
'''
'''
'''
