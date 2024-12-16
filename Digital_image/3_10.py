import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# from matplotlib import pyplot as plt

# 读取图像
img = cv.imread(r"D:\Code\Python\class02\CH3\03160.jpg")
# 灰度变换
grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 生成高斯噪声
noiseGauss = np.random.normal(0, 11, grayimg.shape).astype(np.uint8)
# 创建高斯噪声污染的图像
gaussImg = cv.add(grayimg, noiseGauss)

# 生成椒盐噪声
density = 0.05  # 调整噪声密度
noisePepper = np.random.choice([0, 255], grayimg.shape, True, [1 - density, density])
# 创建椒盐噪声污染的图像
pepperImg = cv.add(grayimg, noisePepper.astype(np.uint8))

plt.subplot(241)
plt.imshow(grayimg, cmap="gray", vmin=0, vmax=255)
plt.axis('off'), plt.title("1. Original")

plt.subplot(242)
plt.imshow(gaussImg, cmap="gray", vmin=0, vmax=255)
plt.axis('off'), plt.title("2. gaussImg")

plt.subplot(243)
plt.imshow(pepperImg, cmap="gray", vmin=0, vmax=255)
plt.axis('off'), plt.title("3. pepperImg")

plt.show()
