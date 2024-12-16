import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv.imread(r"D:\Code\Python\class02\CH3\06007.tif")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
HSI = cv.cvtColor(img, cv.COLOR_RGB2HSV)
# 分割通道
h, s, i = cv.split(HSI)
i = cv.equalizeHist(i)

hsiimg = cv.merge([h, s, i])

# 模型转换
HSI = cv.cvtColor(HSI, cv.COLOR_HSV2RGB)
hsiimg = cv.cvtColor(hsiimg, cv.COLOR_HSV2RGB)

plt.subplot(131), plt.imshow(img), plt.title('Original'), plt.axis('off')
plt.subplot(132), plt.imshow(HSI), plt.title('HSI '), plt.axis('off')
plt.subplot(133), plt.imshow(hsiimg), plt.title('HSI equalizeHist'), plt.axis('off')

plt.show()
