import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
img=mpimg.imread('6.jpg')

lum_img = img[100:500,100:500]
# lum_img = img[:,:,0]
imgplot = plt.imshow(lum_img)
imgplot.set_cmap('nipy_spectral')

plt.colorbar()
plt.show()
# plt.hist(lum_img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')