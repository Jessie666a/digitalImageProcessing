import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv.imread(r"D:\Code\Python\class02\CH3\06009.tif")
HSI = cv.cvtColor(img, cv.COLOR_BGR2HSV)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# HSI图像通道分割
H, S, I = cv.split(HSI)
# 二值化饱和度
ret, binary = cv.threshold(S, np.max(S) * 0.34, 1, cv.THRESH_BINARY)

# 和色调图像相乘
mask = binary * H
# 直方图统计
# 二值化结果
ret, out = cv.threshold(mask, 160, 1, cv.THRESH_BINARY)
# 调整模板为3通道
out3 = cv.merge([out, out, out])
# 和原图像相乘
outimg = out3 * img

plt.subplot(241), plt.imshow(img), plt.title('Original'), plt.axis('off')
plt.subplot(242), plt.imshow(H, cmap='gray'), plt.title('H'), plt.axis('off')
plt.subplot(243), plt.imshow(S, cmap='gray'), plt.title('S'), plt.axis('off')
plt.subplot(244), plt.imshow(I, cmap='gray'), plt.title('I'), plt.axis('off')
plt.subplot(245), plt.imshow(binary, cmap='gray'), plt.title('binary'), plt.axis('off')
plt.subplot(246), plt.imshow(mask, cmap='gray'), plt.title('mask'), plt.axis('off')
plt.subplot(247), plt.hist(mask.ravel(), 256, [0, 255]), plt.title('mask histgram')
plt.subplot(248), plt.imshow(outimg), plt.title('out'), plt.axis('off')

plt.show()
