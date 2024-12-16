import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# from matplotlib import pyplot as plt

# 读取图像
img = cv.imread(r"D:\Code\Python\class02\CH3\03165.tif")
# 灰度变换
grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 中值滤波
median = cv.medianBlur(grayimg, 3)

plt.subplot(121)
plt.imshow(grayimg, cmap="gray", vmin=0, vmax=255)
plt.axis('off'), plt.title("1. Original")

plt.subplot(122)
plt.imshow(median, cmap="gray", vmin=0, vmax=255)
plt.axis('off'), plt.title("2. median")

plt.show()
