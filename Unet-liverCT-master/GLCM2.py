# -*- coding: utf-8 -*-
"""
提取glcm纹理特征
"""
import numpy as np
import cv2
import os
from sklearn.metrics.cluster import entropy
import matplotlib.pyplot as plt
# from skimage.feature import greycomatrix, greycoprops
import skimage.feature
from xlwt import Workbook
def get_inputs(s): # s为图像路径
    input = cv2.imread(s, cv2.IMREAD_GRAYSCALE) # 读取图像，灰度模式
    # 得到共生矩阵，参数：图像矩阵，距离，方向，灰度级别，是否对称，是否标准化
    glcm = skimage.feature.greycomatrix(input, [2, 8, 16], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 256, symmetric=True, normed=True)
    #对比度，想关，能量，同质性
    # for prop in {'contrast', 'dissimilarity','homogeneity', 'energy', 'correlation', 'ASM'}:
    print("-----------0度----------")
    # temp = greycoprops(glcm, prop="contrast")
    # print(max(temp[0]))
    # print(temp[0])
    # print(temp[1])
    # print(temp[2])

    temp = skimage.feature.greycoprops(glcm, prop= "energy")
    # if max(temp[0]==1) or max(temp[1]==1) or max(temp[2]==1):
    print(max(temp[0]),max(temp[1]),max(temp[2]))
    print(min(temp[0]),min(temp[1]),min(temp[2]))
    print(temp[0])
    print(temp[1])
    print(temp[2])
flag = 0
if __name__ == '__main__':
    path = "test/mask/"
    dir = os.listdir(path)
    dir.sort()
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1')
    for i in dir:
        # if flag < 60:
            if i==".DS_Store":
                continue
            print(path+i)
            input = cv2.imread(path+i, cv2.IMREAD_GRAYSCALE)
            glcm = skimage.feature.greycomatrix(input, [1], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 256, symmetric=True,normed=True)
            # 对比度，差异性，同质性，能量，相关，二阶矩
            # for prop in {'contrast', 'dissimilarity','homogeneity', 'energy', 'correlation', 'ASM'}:
            contrast = skimage.feature.greycoprops(glcm, "contrast")
            dissimilarity = skimage.feature.greycoprops(glcm, "dissimilarity")
            homogeneity = skimage.feature.greycoprops(glcm, "homogeneity")
            energy = skimage.feature.greycoprops(glcm, "energy")
            correlation = skimage.feature.greycoprops(glcm, "correlation")
            ASM = skimage.feature.greycoprops(glcm, "ASM")

            entropyImg = entropy(input)#得到熵
            sheet1.write(flag+1,0,i)
            sheet1.write(flag+1,1,contrast[0][0])
            sheet1.write(flag+1,2,dissimilarity[0][0])
            sheet1.write(flag+1,3,homogeneity[0][0])
            sheet1.write(flag+1,4,energy[0][0])
            sheet1.write(flag+1,5,correlation[0][0])
            sheet1.write(flag+1,6,ASM[0][0])
            sheet1.write(flag+1,7,entropyImg)
            print(contrast[0][0],dissimilarity[0][0],homogeneity[0][0],energy[0][0],correlation[0][0],ASM[0][0],entropyImg)

            print("----------分界线-----------")
            flag+=1


wb.save("data/glcm.xls")
