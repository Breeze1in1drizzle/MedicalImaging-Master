from PIL import Image
from PIL import ImageEnhance
from skimage import data, exposure
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
path = "test/mask/"
files =os.listdir(path) #膀胱区域图片
files.sort()
flag = 0
dir = "test/image/"  #原图
files2 = os.listdir(dir)
files2.sort()
'''均值，方差，歪斜度，峰态，能量，熵'''

for f in files:
    if f==".DS_Store":
        continue
    from matplotlib import pyplot as plt
    import cv2
    import numpy as np

    img = cv2.imread(path+f)
    import cv2 as cv
    import numpy as np
    from matplotlib import pyplot as plt

    img = cv.imread('dark.jpg', 0)

    # create a mask  应用遮罩
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[100:300, 100:400] = 255
    masked_img = cv.bitwise_and(img, img, mask=mask)

    # Calculate histogram with mask and without mask
    # Check third argument for mask
    """
    蒙板（掩模） 
    将要统计的部分设置成白色，其他部分设置成黑色，就构成了蒙板。然后传递给原图像即可
    """
    hist_full = cv.calcHist([img], [0], None, [256], [0, 256])
    hist_mask = cv.calcHist([img], [0], mask, [256], [0, 256])

    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.imshow(mask, 'gray')
    plt.subplot(223), plt.imshow(masked_img, 'gray')
    plt.subplot(224), plt.plot(hist_full, color='r'), plt.plot(hist_mask, color='b')
    plt.xlim([0, 256])
    plt.tight_layout()
    plt.show()
    break