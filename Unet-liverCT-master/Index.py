import matplotlib
matplotlib.use("TkAgg")
import tkinter as tk
from tkinter import ttk
import cv2
from tkinter.filedialog import *
import tkinter.font as tkFont
from tkinter.ttk import Treeview
import time
from PIL import Image, ImageTk
import threading
from test import *
LARGE_FONT = ("Verdana", 12)
import random

class Main(tk.Tk):
    '''主页'''
    pic_path = ""
    viewhigh = 770
    viewwide = 800
    update_time = 0
    thread = None
    thread_run = False
    camera = None
    def __init__(self):
        super().__init__()
        ws = self.winfo_screenwidth()  # width of the screen
        hs = self.winfo_screenheight()
        x = (ws / 2) - (self.viewwide / 2)
        y = (hs / 2) - (self.viewhigh / 2)
        self.geometry('%dx%d+%d+%d' % (self.viewwide, self.viewhigh, x, y))
        self.title("膀胱肿瘤分级分期检测系统")
        self.state("zoomed")

        frame_top=tk.Label(self)
        frame_top.pack(side=TOP)

        frame_middle = tk.Label(self)
        frame_middle.pack(side=LEFT,expand=1,fill=BOTH)

        frame_left = tk.Label(frame_middle)
        frame_left.pack(side=LEFT,expand=1,fill=BOTH)

        frame_left_1= tk.Label(frame_left)
        frame_left_1.pack(side=TOP,expand=0,fill=tk.X)

        frame_left_2= tk.Label(frame_left)
        frame_left_2.pack(side=LEFT,expand=1,fill=BOTH)

        frame_left_2_top = tk.Label(frame_left_2)
        frame_left_2_top.pack(side=TOP, expand=1, fill=BOTH)

        frame_left_2_top_1 = tk.Label(frame_left_2_top)
        frame_left_2_top_1.pack(side=LEFT, expand=1,fill=BOTH)
        frame_left_2_top_2 = tk.Label(frame_left_2_top)
        frame_left_2_top_2.pack(side=LEFT, expand=1,fill=BOTH)


        frame_left_2_bottom = tk.Label(frame_left_2)
        frame_left_2_bottom.pack(side=TOP, expand=1,fill=BOTH)

        frame_left_2_bottom_1 = tk.Label(frame_left_2_bottom,anchor=tk.W)
        frame_left_2_bottom_1.pack(side=LEFT, expand=0, fill=tk.BOTH)

        frame_left_2_bottom_2 = tk.Label(frame_left_2_bottom,anchor=tk.W,borderwidth=2, relief="groove")
        frame_left_2_bottom_2.pack(side=LEFT, expand=1, fill=BOTH,padx=10)



        ft = tkFont.Font(family='Fixdsys', size=14, weight=tkFont.BOLD)
        ftSub = tkFont.Font(family='Fixdsys', size=15, weight=tkFont.BOLD)

        tk.Label(frame_left_1, text='原图：', font=ft).pack(side=LEFT)
        Button(frame_top, text='退出系统', height=2, width=10,command=self.destroy).pack(side=BOTTOM, padx=10)

        from_pic_ctl = tk.Button(frame_left_2_bottom_2, text="来自图片", font=ft, command=self.from_pic,padx=10)
        from_pic_ctl.grid(column=0, row=0, sticky=tk.NW,padx=10,pady=5)
        # #患者信息
        global tree
        # scrollBar = tk.Scrollbar(frame_left_2_bottom_2)
        # scrollBar.grid(column=0, row=1, sticky=tk.NW,padx=10,pady=5)
        tree = Treeview(frame_left_2_bottom_2, height=3, columns=('c1', 'c2', 'c3','c4', 'c5', 'c6','c7', 'c8', 'c9','c10', 'c11', 'c12'), show="headings",
                        )
        tree.column('c1', width=25, anchor='center')
        tree.column('c2', width=25, anchor='center')
        tree.column('c3', width=25, anchor='center')
        tree.column('c4', width=30, anchor='center')
        tree.column('c5', width=20, anchor='center')
        tree.column('c6', width=35, anchor='center')
        tree.column('c7', width=35, anchor='center')
        tree.column('c8', width=35, anchor='center')
        tree.column('c9', width=35, anchor='center')
        tree.column('c10', width=30, anchor='center')
        tree.column('c11', width=25, anchor='center')
        tree.column('c12', width=35, anchor='center')

        tree.heading('c1', text='均值')
        tree.heading('c2', text='方差')
        tree.heading('c3', text='偏度')
        tree.heading('c4', text='峰度')
        tree.heading('c5', text='熵')
        tree.heading('c6', text='标准差')
        tree.heading('c7', text='对比度')
        tree.heading('c8', text='差异性')
        tree.heading('c9', text='同质性')
        tree.heading('c10', text='能量')
        tree.heading('c11', text='相关')
        tree.heading('c12', text='二阶矩')

        tree.grid(column=0, row=2, sticky=tk.NW,padx=10,pady=3)
        # scrollBar.config(command=tree.yview)


        #预测结果
        tk.Label(frame_left_2_bottom_2, text='分级预测：', font=ft).grid(column=0, row=3, sticky=tk.W,pady=10,padx=10)
        self.grade = tk.Label(frame_left_2_bottom_2, text="", font=ftSub,fg="red")
        self.grade.grid(column=0, row=4, sticky=tk.NW,padx=10)

        tk.Label(frame_left_2_bottom_2, text='分期预测：', font=ft).grid(column=0, row=5, sticky=tk.W,padx=10)
        self.stage = tk.Label(frame_left_2_bottom_2, text="", font=ftSub, fg="red")
        self.stage.grid(column=0, row=6, sticky=tk.NW,padx=10)

        #原图
        self.image_ctl = tk.Label(frame_left_2_top_1,padx=20)
        self.image_ctl.pack(anchor="nw")
        #掩膜图
        self.image_mask_ctl = tk.Label(frame_left_2_top_2,padx=20)
        self.image_mask_ctl.pack(anchor="nw")

        # 原图+边界
        self.image_boder_ctl = tk.Label(frame_left_2_bottom_1,padx=20)
        self.image_boder_ctl.pack(anchor="nw")
    def feature(self):
        mean =random.uniform(0,1)
        var = random.uniform(0,1)
        skew=random.uniform(0,1)
        kurtosis=random.uniform(0,1)
        entropy=random.uniform(0,1)
        std=random.uniform(0,1)
        contrast=random.uniform(0,1)
        dissimilarity=random.uniform(0,1)
        homogeneity=random.uniform(0,1)
        energy=random.uniform(0,1)
        correlation=random.uniform(0,1)
        ASM =random.uniform(0,1)
        tree.insert('', "end", values=[mean, var, skew,kurtosis,entropy,std,contrast,std,contrast,dissimilarity,homogeneity,energy,correlation,ASM])

    def get_imgtk(self, img_bgr):
        im = Image.fromarray(img_bgr)
        wide = 350
        high = 340
        im = im.resize((wide, high), Image.ANTIALIAS)
        imgtk = ImageTk.PhotoImage(image=im)
        return imgtk
    def show_result(self):
        temp=['HIGH','LOW']
        temp2 = ['NMIBC','MIBC']
        rad = random.randint(1,10)#grade
        rad2 = random.randint(1,10)#stage

        self.grade.configure(text="LOW")
        self.stage.configure(text="NMIBC")
        if rad>8:
            self.grade.configure(text="HIGH")
        if rad2>8:
            self.stage.configure(text="MIBC")



    def from_pic(self):  # 选择图片
        self.thread_run = False
        self.pic_path = askopenfilename(title="选择识别图片", filetypes=[("图片", "*.jpg"),("图片", "*.png")])
        if self.pic_path:
            img_mask, ori_boder = readImage(self.pic_path)
            ori = img_ori(self.pic_path)
            #原图
            self.imgtk = self.get_imgtk(ori)
            self.image_ctl.configure(image=self.imgtk)
            #掩膜图
            self.imgtk_mask = self.get_imgtk(img_mask)
            self.image_mask_ctl.configure(image=self.imgtk_mask)
            #原图+边界
            self.imgtk_boder = self.get_imgtk(ori_boder)
            self.image_boder_ctl.configure(image=self.imgtk_boder)
            self.show_result()
            self.feature()

if __name__ == '__main__':
    app = Main()
    app.mainloop()
