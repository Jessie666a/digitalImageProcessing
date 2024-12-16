import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv.imread(r"D:\Code\Python\class02\CH3\06010.tif")
HSI = cv.cvtColor(img, cv.COLOR_BGR2HSV)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 颜色上下限
lowerColor = (100, 100, 100)
upperColor = (124, 255, 255)

# 模板结果
binary = cv.inRange(HSI, lowerColor, upperColor)

# 应用掩膜将蓝色背景替换为红色
mask = cv.bitwise_not(binary)
matting = cv.bitwise_or(img, img, mask=mask)
# 背景颜色替换
replace = matting.copy()
replace[mask == 0] = [255, 0, 0]

# 显示
plt.subplot(221), plt.imshow(img), plt.title('Original'), plt.axis('off')
plt.subplot(222), plt.imshow(binary, cmap='gray'), plt.title('mask'), plt.axis('off')
plt.subplot(223), plt.imshow(matting), plt.title('matting'), plt.axis('off')
plt.subplot(224), plt.imshow(replace), plt.title('out'), plt.axis('off')

plt.show()
