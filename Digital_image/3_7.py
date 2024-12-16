import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# from matplotlib import pyplot as plt

# 读取图像
img = cv.imread(r"E:\Digital_images\image\image\CH3\03165.tif")
# 灰度变换
grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 盒式滤波
ksize = (5, 5)
# 方式1
box1 = cv.blur(grayimg, ksize)
# 方式2
# 创建归一化盒式滤波器 核
kernel = np.ones(ksize, np.float32) / (ksize[0] * ksize[1])
box2 = cv.filter2D(grayimg, -1, kernel)
# 方式3
ksize = (21, 21)
box3 = cv.boxFilter(grayimg, -1, ksize, True)

plt.subplot(221)
plt.imshow(grayimg, cmap="gray", vmin=0, vmax=255)
plt.axis('off'), plt.title("1. Original")

plt.subplot(222)
plt.imshow(box1, cmap="gray", vmin=0, vmax=255)
plt.axis('off'), plt.title("2. box1")

plt.subplot(223)
plt.imshow(box2, cmap="gray", vmin=0, vmax=255)
plt.axis('off'), plt.title("4. box2")

plt.subplot(224)
plt.imshow(box3, cmap="gray", vmin=0, vmax=255)
plt.axis('off'), plt.title("5. box3")

plt.show()
cv.waitKey(0)
