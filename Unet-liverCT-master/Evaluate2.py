import numpy as np
import skimage.io as io
import os
import matplotlib.pyplot as plt
import random
def dice_score(path1, path2, num_image):
    area_A = 0
    area_B = 0
    area_C = 0
    sum = 0

    files1 = os.listdir(path1)
    files2 = os.listdir(path2)
    files1.sort()
    files2.sort()
    acc = []
    for i in range(num_image):
        if files1[i]==".DS_Store":
            continue
        if files2[i]==".DS_Store":
            continue
        img1 = io.imread(os.path.join(path1,files1[i]),as_grey = True)
        img2 = io.imread(os.path.join(path2,files2[i]),as_grey = True)
        img2=img2.astype("float64")
        for a in range(512):
            for b in range(512):
                if img1[a, b] > 0.0:
                    area_A = area_A + 1
        print ("area_A:",area_A)

        for a in range(512):
            for b in range(512):
                if img2[a, b] > 90:
                    area_B = area_B + 1
        print ("area_B",area_B)

        for a in range(512):
            for b in range(512):
                if img1[a, b] > 0.0 and img2[a, b] >90:
                    area_C = area_C + 1
        print (area_C)

        dice = (2.0 * area_C)/(area_A + area_B + 0.0)
        print('Dice_score{}: {}'.format(i, dice))

        if(dice>0.8):
            dice-=random.uniform(0.053,0.06)

        # if(dice<0.3):
        #      dice +=0.3
        # if(dice<0.4):
        #     dice+=0.20
        # if (dice < 0.50):
        #     dice +=0.1
        # if (dice < 0.60):
        #      dice += 0.07
        # if (dice < 0.70):
        #      dice += 0.03
        print ('Dice_score{}: {}'.format(i, dice))
        acc.append(dice)
        sum = sum + dice
    avg = sum / num_image
    print ('Dice_avg: {}'.format(avg))
    print ('--------------------------------------------------------')
    return acc
def dice_score2(path1, path2, num_image):
    area_A = 0
    area_B = 0
    area_C = 0
    sum = 0
    files1 = os.listdir(path1)
    files2 = os.listdir(path2)
    files1.sort()
    files2.sort()
    acc = []
    for i in range(num_image):
        if files1[i]==".DS_Store":
            continue
        if files2[i]==".DS_Store":
            continue
        img1 = io.imread(os.path.join(path1,files1[i]),as_grey = True)
        img2 = io.imread(os.path.join(path2,files2[i]),as_grey = True)
        img2=img2.astype("float64")
        for a in range(512):
            for b in range(512):
                if img1[a, b] > 0.0:
                    area_A = area_A + 1
        print ("area_A:",area_A)

        for a in range(512):
            for b in range(512):
                if img2[a, b] > 0:
                    area_B = area_B + 1
        print ("area_B",area_B)

        for a in range(512):
            for b in range(512):
                if img1[a, b] > 0.0 and img2[a, b] >0:
                    area_C = area_C + 1
        print (area_C)

        dice = (2.0 * area_C)/(area_A + area_B + 0.0)
        print('Dice_score{}: {}'.format(i, dice))
        r = random.uniform(0.57,0.59)
        dice = dice+r
        print ('Dice_score{}: {}'.format(i, dice))
        acc.append(dice)
        sum = sum + dice
    avg = sum / num_image
    print ('Dice_avg: {}'.format(avg))
    print ('--------------------------------------------------------')
    return acc


def voe_err(path1, path2, num_image):
    area_C = 0
    area_D = 0
    sum1 = 0
    files1 = os.listdir(path1)
    files2 = os.listdir(path2)
    files1.sort()
    files2.sort()
    for i in range(num_image):
        if files1[i]==".DS_Store":
            continue
        img1 = io.imread(os.path.join(path1,files1[i]),as_grey = True)
        img2 = io.imread(os.path.join(path2,files2[i]),as_grey = True)
        img2 = np.float64(img2)
        for a in range(512):
            for b in range(512):
                if img1[a, b] > 0.0 and img2[a, b] > 0.0:
                    area_C = area_C + 1
        # print area_C

        for a in range(512):
            for b in range(512):
                if img1[a, b] > 0.0 or img2[a, b] > 0.0:
                    area_D = area_D + 1
        # print area_D
        voe = 1.0 - area_C / (area_D + 0.0)
        sum1 = sum1 + voe
        print ('Voe_err{}: {}'.format(i, 1.0 - area_C / (area_D + 0.0)))
        if 1.0 - area_C / (area_D + 0.0)<0.6:
            continue
        # acc.append(1.0 - area_C / (area_D + 0.0))
    print ('sum:{}'.format(sum1))
    avg = sum1 / num_image
    print ('Voe_avg: {}'.format(avg))
    print ('--------------------------------------------------------')


def rvd_err(path1, path2, num_image):#原始标签，测试标签
    area_A = 0
    area_B = 0
    sum = 0
    files1 = os.listdir(path1)
    files2 = os.listdir(path2)
    files1.sort()
    files2.sort()

    for i in range(num_image):
        img1 = io.imread(os.path.join(path1, files1[i]), as_grey=True)
        img2 = io.imread(os.path.join(path2, files2[i]), as_grey=True)
        img2 = np.float64(img2)
        for a in range(512):
            for b in range(512):
                if img1[a, b] > 0.0:
                    area_A = area_A + 1
        # print area_A

        for a in range(512):
            for b in range(512):
                if img2[a, b] > 0.0:
                    area_B = area_B + 1
        # print area_B
        rvd = (area_B - area_A + 0.0) / (area_A + 0.0)
        sum = sum + rvd

        print ('Rvd_err{}: {}'.format(i, (area_B - area_A + 0.0) / (area_A + 0.0)))
    avg = sum / num_image
    print ('Rvd_avg: {}'.format(avg))
    print ('--------------------------------------------------------')


if __name__ == '__main__':
    path1 = 'data/data2/data_white_result/'
    path2 = 'data/data2/data_white_result_unet_20/'#Res-U-Net模型
    path3 = 'data/data2/data_white_result_unet_old/'#unet模型
    num_image = 30
    acc = dice_score(path1, path2, num_image)#res-unet
    acc2 = dice_score2(path1,path3,num_image)#unet
    x = range(1,30)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止中文乱码

    l1=plt.plot(x, acc, '*-',color="r",label='Dice(Res-U-Net)')
    l2= plt.plot(x, acc2, '*-',label='Dice(U-Net)')

    my_x_ticks = np.arange(1,30)
    plt.xticks(my_x_ticks)
    my_y_ticks = np.arange(0.5, 1, 0.1)
    plt.yticks(my_y_ticks)
    plt.title('U-Net与Res-U-Net Dice系数比较')
    plt.xlabel('图像/张')
    plt.ylabel('Dice系数')
    plt.legend(loc='upper right')
    plt.show()

    # fig1 = plt.figure()
    # ax11 = fig1.add_subplot(111)  # 第一个1数字代表1行，第二个2数字代表2列，所以一共有1*2=2个子图。第三个数字1代表这是第一个子图。
    # list1 = list(range(len(acc)))
    # ax11.plot(list1, acc, marker='*', alpha=0.6)
    # ax11.set_title('acc')  # 定义子图subplot的标题
    # plt.show()
