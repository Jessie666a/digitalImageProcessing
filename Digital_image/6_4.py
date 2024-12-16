import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv.imread(r"E:\Digital_images\image\image\CH3\06004.tif")

# 伽马变换
gamaimg = np.power(img, 1.5)

# 归一化处理
gamaimg = gamaimg - np.min(gamaimg)  # 左侧范围调整为0
gamaimg = gamaimg / np.max(gamaimg)  # 右侧范围调整为1

# 拉伸对比度
med = np.median(gamaimg)  # 查找中值
Simg = 1 / (1 + np.power(med / (gamaimg + 1e-8), 4.5))  # S变换
Simg = Simg - np.min(Simg)  # 左侧范围调整为0
Simg = Simg / np.max(Simg)  # 右侧范围调整为1

# 计算S曲线图数据
x1 = np.linspace(Simg.min(), Simg.max(), num=200)
y1 = 1 / (1 + np.power((med / (x1 + 1e-8)), 4.5))  # 对比度拉伸函数

# 调整为RGB图像用于plt 显示
Simg = (Simg * 255).astype(np.uint8)  # 变换范围到[0,255]
gamaimg = (gamaimg * 255).astype(np.uint8)  # 变换范围到[0,255]

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gamaimg = cv.cvtColor(gamaimg, cv.COLOR_BGR2RGB)
Simg = cv.cvtColor(Simg, cv.COLOR_BGR2RGB)

# 循环显示
imgList = [img, gamaimg, Simg,
           ]
imgTitle = ['1. Original', '2. gama ', '3. S ',
            ]
imgStart = 221  # 子图起始位置，小于10幅子图时可用
imgNum = 3  # 子图数量，对应以上两个列表长度
for i in range(imgNum):
    plt.subplot(imgStart + i)
    plt.imshow(imgList[i], cmap='gray')
    plt.axis('off')
    plt.title(imgTitle[i])

plt.subplot(224), plt.plot(x1, y1)

plt.show()
