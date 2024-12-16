import cv2 as cv
import numpy as np

# 加载图像
lena = cv.imread(r"E:/Digital_images/image/image/lena.jpg")

# 色彩转换，彩色图像转换为灰度图
# 灰度图不包含色彩信息
grayLena = cv.cvtColor(lena, cv.COLOR_BGR2GRAY)

''''''
# 创建最大灰度矩阵
grayMax = np.ones((grayLena.shape[1], grayLena.shape[0]), dtype=np.uint8) * 255
# 反色
invGray = cv.subtract(grayMax, grayLena)

invGray2 = cv.bitwise_not(grayLena)
# 创建彩色图像最大灰度矩阵
lenaMax = np.ones((lena.shape[1], lena.shape[0], lena.shape[2]), dtype=np.uint8) * 255
# 反色
invLena = cv.subtract(lenaMax, lena)

result = cv.subtract(invGray2, invGray)

print(lena.shape)
print(grayLena.shape)

cv.imshow('image', lena)
cv.imshow('gray image', grayLena)
cv.imshow('inv image', invGray)
cv.imshow('invGray2', invGray2)
cv.imshow('result', result)
cv.waitKey(0)
