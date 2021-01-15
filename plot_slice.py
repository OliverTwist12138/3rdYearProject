
from matplotlib import pyplot as plt
import numpy as np

images = np.load('./validation/apnea/PT17A6_During_10.npy')
image = images[:,:,10]
print(image.dtype)

plt.imshow(image,norm=True)
plt.show()
