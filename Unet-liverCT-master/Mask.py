# 导入库
import numpy as np
import argparse
from matplotlib import pyplot as plt
from scipy import stats
from statistics import stdev
import math
import cv2
import os
from xlwt import Workbook
wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1')
path = "test/image/"
dir  = "test/mask/"
files = os.listdir(dir)  #获得掩膜图片白色
files.sort()
flag =0
for i in (files):
    # if flag<3:
        if i==".DS_Store":
            continue
        image = cv2.imread(path + i)
        mask = cv2.imread(dir+i)
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        for j in range(512):
            for k in range(512):
                if mask[j][k]!=0:
                    mask[j][k]=255
        print(path+i)
        masked = cv2.bitwise_and(image, image, mask=mask)
        hist_full = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_mask = cv2.calcHist([image], [0], mask, [256], [0, 256])
        # plt.subplot(221), plt.imshow(image, 'gray')
        # plt.subplot(222), plt.imshow(mask, 'gray')
        # plt.subplot(223), plt.imshow(masked, 'gray')
        # plt.subplot(224), plt.plot(hist_full, color='r'), plt.plot(hist_mask, color='b')
        # plt.xlim([0, 256])
        # plt.tight_layout()
        # plt.show()
        '''shb'''
        print(flag,hist_mask.mean(),hist_mask.var(),stats.skew(hist_mask)[0],stats.kurtosis(hist_mask)[0]
              ,stats.entropy(hist_mask)[0],np.std(hist_mask))

        sheet1.write(flag + 1, 0, i)
        sheet1.write(flag+1,1,  str(hist_mask.mean()))#均值
        sheet1.write(flag + 1, 2, str(hist_mask.var()))#方差
        sheet1.write(flag + 1, 3, str(stats.skew(hist_mask)[0]))#偏度
        sheet1.write(flag + 1, 4, str(stats.kurtosis(hist_mask)[0]))#峰度
        sheet1.write(flag + 1, 5, str(stats.entropy(hist_mask)[0]))#熵
        sheet1.write(flag + 1, 6, str(np.std(hist_mask)))#标准差
        flag+=1
    # cv2.imwrite("test/mask/"+images[int(img_index)],masked)

        #计算GLCM特征

wb.save("data/mean.xls")
