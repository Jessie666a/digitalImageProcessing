import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# from matplotlib import pyplot as plt

# 读取图像
img = cv.imread(r"D:\Code\Python\class02\CH3\03165.tif")
# 灰度变换
grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 方式1
ksize = (5, 5)
gauss1 = cv.GaussianBlur(grayimg, ksize, 0)
# 方式2
# 1D高斯核
kernel = cv.getGaussianKernel(5, 1)
# 2D高斯核
kernel = kernel.T * kernel
#
gauss2 = cv.filter2D(grayimg, -1, kernel)

gauss3 = cv.GaussianBlur(grayimg, (21, 21), 3.5)
gauss4 = cv.GaussianBlur(grayimg, (43, 43), 7)

plt.subplot(231)
plt.imshow(grayimg, cmap="gray", vmin=0, vmax=255)
plt.axis('off'), plt.title("1. Original")

plt.subplot(232)
plt.imshow(gauss1, cmap="gray", vmin=0, vmax=255)
plt.axis('off'), plt.title("2. gauss1")

plt.subplot(233)
plt.imshow(gauss2, cmap="gray", vmin=0, vmax=255)
plt.axis('off'), plt.title("3. gauss2")

plt.subplot(234)
plt.imshow(gauss3, cmap="gray", vmin=0, vmax=255)
plt.axis('off'), plt.title("4. gauss3,size = 21\n$\sigma=3.5$")

plt.subplot(235)
plt.imshow(gauss4, cmap="gray", vmin=0, vmax=255)
plt.axis('off'), plt.title("5. gauss4,size = 43\n$\sigma=7$")
plt.show()

cv.waitKey(0)
