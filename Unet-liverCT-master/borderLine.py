from PIL import Image
from PIL import ImageEnhance
from skimage import data, exposure
import cv2
import numpy as np
import tqdm
import os

'''膀胱区域的图像边界更加清晰'''
path = "test/mask/"
files = os.listdir(path)
files.sort()
flag = 0
dir = "test/image/"
files2 = os.listdir(dir)
files2.sort()
for f in files:
    # if flag<300:
    if f == ".DS_Store":
        continue
    img = cv2.imread(path + f)  # 已经得到的掩膜图
    num = f[:-4]
    ori = cv2.imread(dir + f)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    image, contours, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if (cv2.contourArea(cnt) > 2000)]  # 面积小于200的排除掉
    cv2.drawContours(ori, contours, -1, (0, 0, 255), 1)
    print(len(contours))
    if len(contours)==0:
        continue
    cv2.imwrite("border/"+f, ori)
# flag+=1
