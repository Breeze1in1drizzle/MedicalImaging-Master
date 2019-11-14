from PIL import Image
from PIL import ImageEnhance
from skimage import data, exposure
import cv2
import numpy as np
import tqdm
import os
import skimage.io as io
def simgelImage(path):
    print(path)
    img = cv2.imread("newTest/label/"+path)  # 已经得到的掩膜图
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # cv2.imwrite("test2/re1.jpg",img)
    for i in range(512):
        for j in range(512):
            if img[i][j] >50:
                img[i][j] = 255
            else:
                img[i][j] = 0
    olding = img  # 对应的灰度图
    # cv2.imwrite("test2/re2.jpg", img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    clode = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    image, contours, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        temp = cv2.contourArea(cnt)
        if temp < 3000:
            x, y, w, h = cv2.boundingRect(cnt)
            for o in range(x, x + w):
                for k in range(y, y + h):
                    olding[k][o] = 0  # 排除小个区域
    contours = [cnt for cnt in contours if (cv2.contourArea(cnt) > 2000)]  # 面积小于200的排除掉
    tempRadius = []
    tempPoints = []
    # cv2.drawContours(ori, contours, -1, (255, 0, 0), 1)
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)  # 得到中心点和半径
        tempRadius.append(int(radius))
        tempPoints.append([x, y])
    cv2.imwrite("newTest/re/"+path, olding)
    # cv2.imwrite("test2/ori.jpg",ori)
    # return olding,ori
dir = "newTest/label/"
files = os.listdir(dir)
files.sort()
for i in files:
    print(i)
    if i==".DS_Store":
        continue
    simgelImage(i)
# path = "test/label/"
# files =os.listdir(path)
# files.sort()
# flag = 0
# dir = "test/image/"
# files2 = os.listdir(dir)
# files2.sort()
# for f in files:
#         if str(f)==".DS_Store":
#             continue
#         img = cv2.imread(path+f)#已经得到的掩膜图
#         print(f)
#         num = f[:-4]
#         ori = cv2.imread(dir+files2[int(num)])
#         img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#         for i in range(512):
#             for j in range(512):
#                 if img[i][j]>50:
#                    img[i][j]=255
#                 else:
#                     img[i][j]=0
#         olding = img  #对应的灰度图
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
#         clode = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#         image, contours, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         cutImage = []
#         for cnt in contours:
#             temp = cv2.contourArea(cnt)
#             if temp<3000:
#                 x, y, w, h = cv2.boundingRect(cnt)
#                 for o in range(x,x+w):
#                     for k in range(y,y+h):
#                         olding[k][o]=0#排除小个区域
#         contours = [cnt for cnt in contours if (cv2.contourArea(cnt) > 2000)]  # 面积小于200的排除掉
#         tempList = []
#         tempRadius = []
#         tempPoints = []
#         cv2.drawContours(ori, contours, -1, (0, 0, 255), 1)
#         if len(contours)==0:
#             continue
#         for cnt in contours:
#             # x, y, w, h = cv2.boundingRect(cnt)
#             (x, y), radius = cv2.minEnclosingCircle(cnt)#得到中心点和半径
#             tempRadius.append(int(radius))
#             tempPoints.append([x,y])
#         print(num,files2[int(num)])
#         cv2.imwrite("temp/"+f,olding)
    # flag+=1
