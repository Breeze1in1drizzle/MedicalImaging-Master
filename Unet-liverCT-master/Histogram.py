from PIL import Image
from PIL import ImageEnhance
from skimage import data, exposure
import cv2
import numpy as np
import tqdm
import os
path = "mask/"
files =os.listdir(path)
files.sort()
flag = 0
dir = "test/image/"
files2 = os.listdir(dir)
files2.sort()
for f in files:
    if flag<30:
        if f==".DS_Store":
            continue
        img = cv2.imread(path+f)#已经得到的掩膜图
        num = f[:-4]
        ori = cv2.imread(dir+f)
        olding = img
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        for i in range(512):
            for j in range(512):
                if img[i][j]>30:
                   img[i][j]=255
                else:
                    img[i][j]=0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        clode = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        image, contours, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cutImage = []
        for cnt in contours:
            temp = cv2.contourArea(cnt)
            if temp<3000:
                x, y, w, h = cv2.boundingRect(cnt)
                for o in range(x,x+w):
                    for k in range(y,y+h):
                        img[k][o]=0#排除小个区域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 150))
        clode = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        print("temp/"+f)
        cv2.imwrite("temp/"+f,clode)
    flag+=1