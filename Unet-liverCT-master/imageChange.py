import os
import cv2
dir = "newTest/"
files = os.listdir(dir)
files.sort()
for i in files:
    print(i)
    if i==".DS_Store":
        continue
    img = cv2.imread(dir+i)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    for j in range(0,512):
        for k in range(0,512):
            if img[j][k]>50:
                img[j][k]=255
            else:
                img[j][k] = 0

    cv2.imwrite("newTest/re/"+i,img)
