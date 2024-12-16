import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# from matplotlib import pyplot as plt

# 读取图像
img = cv.imread(r"D:\Code\Python\class02\CH3\03169.tif")
# 灰度变换
grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 方式1
# 拉普拉斯变换
lapImg = cv.Laplacian(grayimg, cv.CV_64F, ksize=3)
# 缩放运算结果
lapImg1 = lapImg - min(lapImg.ravel())
lapImg1 = ((255 * lapImg1) / max(lapImg1.ravel())).astype(np.uint8)

sharpening = grayimg + (-1) * lapImg
# 缩放结果
sharpening = cv.convertScaleAbs(sharpening)

# 循环显示
imgList = [grayimg, lapImg, lapImg1, sharpening]
imgTitle = ['1. Original', '2. Laplacian', '3. abs Laplacian', '4. sharpening']
imgSub = 221  # 子图起始位置，小于10个子图时可用
imgNum = 4
for i in range(imgNum):
    plt.subplot(imgSub + i)
    plt.imshow(imgList[i], cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.title(imgTitle[i])

plt.show()
