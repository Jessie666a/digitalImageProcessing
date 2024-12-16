import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# from matplotlib import pyplot as plt

# 读取图像
img = cv.imread(r"D:\Code\Python\class02\CH3\03165.tif")
# 灰度变换
grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Sobel 算子
sobelX = cv.Sobel(grayimg, cv.CV_64F, 1, 0, ksize=3)
sobelY = cv.Sobel(grayimg, cv.CV_64F, 0, 1, ksize=3)
sobelXAbs = cv.convertScaleAbs(sobelX)
sobelYAbs = cv.convertScaleAbs(sobelY)

sobel = sobelX + sobelY

# 循环显示
imgList = [grayimg, sobelX, sobelXAbs, sobelY, sobelYAbs, sobel]
imgTitle = ['1. Original', '2. sobelX', '3. sobelX abs',
            '4. sobelY', '5. sobelY abs', '6. sobel']
imgSub = 231  # 子图起始位置，小于10个子图时可用
imgNum = 6
for i in range(imgNum):
    plt.subplot(imgSub + i)
    plt.imshow(imgList[i], cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.title(imgTitle[i])

plt.show()
