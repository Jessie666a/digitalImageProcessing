import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# from matplotlib import pyplot as plt

# 读取图像
img = cv.imread(r"D:\Code\Python\class02\CH3\03168.tif")
# 灰度变换
grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 钝化遮蔽
# 高斯滤波
gaussImg = cv.GaussianBlur(grayimg, (31, 31), 5)
# 钝化模板
mask = cv.subtract(grayimg, gaussImg)
# 锐化结果
rstImg1 = cv.add(grayimg, (0.5 * mask).astype(np.uint8))
rstImg2 = cv.add(grayimg, (1 * mask).astype(np.uint8))
rstImg3 = cv.add(grayimg, (2 * mask).astype(np.uint8))

# 循环显示
imgList = [grayimg, gaussImg, mask, rstImg1, rstImg2, rstImg3]
imgTitle = ['1. Original', '2. gaussImg', '3. mask',
            '4. result \nk=0.5', '5. result \nk=1', '6. result \nk=2']
imgSub = 231  # 子图起始位置，小于10个子图时可用
imgNum = 6
for i in range(imgNum):
    plt.subplot(imgSub + i)
    plt.imshow(imgList[i], cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.title(imgTitle[i])

plt.show()
