import matplotlib.pyplot as plt
import numpy as np
# program name： main.py
y1 = [0.96, 0.65, 0.58, 0.43, 0.41, 0.37, 0.35, 0.32, 0.31, 0.3, 0.298, 0.289, 0.278]
y2 = [0.85, 0.48, 0.36, 0.31, 0.24, 0.22, 0.21, 0.2, 0.18, 0.186, 0.173, 0.183, 0.171]

x = [0,2.5,5,7.5,10,12.5,15,17.5,20,22.5,25,27.5,30]

label2 = range(1,14)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止中文乱码
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(x,y1, 'r')
plt.plot(x,y2, 'b')
my_x_ticks = np.arange(0, 30, 2.5)
plt.xticks(my_x_ticks)
plt.legend(['Dice(U-Net)','Dice(Res-U-Net)'],loc='upper right')
plt.grid()
plt.show()
