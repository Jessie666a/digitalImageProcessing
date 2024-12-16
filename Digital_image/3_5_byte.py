import cv2 as cv  # 导入OpenCV 库
import numpy as np  # 导入Numpy 库
import matplotlib.pyplot as plt

# from matplotlib import pyplot as plt

# 读取图像
lena = cv.imread(r"E:\Digital_images\image\image\lena.jpg")
# 转换色彩空间
grayLena = cv.cvtColor(lena, cv.COLOR_BGR2GRAY)

# 取比特平面8
BIT8 = np.bitwise_and(grayLena, 128) >> 7

# 取比特平面1
BIT1 = np.bitwise_and(grayLena, 1) >> 0

# 绘制绘图原图
plt.subplot(331)
plt.imshow(grayLena, cmap='gray')
plt.axis('off')
plt.title('gray')


# 绘制BIT8
plt.subplot(332)
plt.imshow(BIT8, cmap='gray')
plt.axis('off')
plt.title('BIT8')

# 绘制BIT8
plt.subplot(339)
plt.imshow(BIT1, cmap='gray')
plt.axis('off')
plt.title('BIT1')

plt.show()
cv.waitKey(0)
