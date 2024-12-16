import cv2 as cv
import numpy as np

# 加载图像
lena = cv.imread(r"E:/Digital_images/image/image/lena.jpg")
panda = cv.imread(r"E:/Digital_images/image/image/panda.jpeg")

# 图像缩放，保持图像大小一致，用于后续运算
panda = cv.resize(panda, (lena.shape[1], lena.shape[0]))

print(panda[0, 0])
print(lena[0, 0])
# 图像加法
addImage = cv.add(lena, panda)
addResult = lena + panda
print(addImage[0, 0])
print(addResult[0, 0])
# 图像减法
subImage = cv.subtract(lena, panda)

# 图像乘法
multiplyImage = cv.multiply(lena, panda)

# 图像除法
divideImage = cv.divide(lena, panda)

# 显示图像
cv.imshow('add', addImage)
cv.imshow('add numpy', addResult)

# cv.imshow('subtract',subImage)
# cv.imshow('multiplyImage',multiplyImage)
# cv.imshow('divideImage',divideImage)

# 等待按键操作
cv.waitKey(0)
