import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv.imread(r"D:\Code\Python\class02\CH3\06009.tif")
HSI = cv.cvtColor(img, cv.COLOR_BGR2HSV)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 颜色上下限
lowerColor = (156, 43, 46)
upperColor = (180, 255, 255)

# 模板结果
mask = np.uint8(cv.inRange(HSI, lowerColor, upperColor) / 255)
# 扩展模板为三通道
mask3 = cv.merge([mask, mask, mask])

out = mask3 * img

# 显示
plt.subplot(131), plt.imshow(img), plt.title('Original'), plt.axis('off')
plt.subplot(132), plt.imshow(mask, cmap='gray'), plt.title('mask'), plt.axis('off')
plt.subplot(133), plt.imshow(out), plt.title('out'), plt.axis('off')

plt.show()
