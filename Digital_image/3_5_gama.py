import cv2 as cv  # 导入OpenCV 库
import numpy as np  # 导入Numpy 库
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
# 读取图像
lena = cv.imread(r"E:\Digital_images\image\image\lena.jpg")
# 转换色彩空间
grayLena = cv.cvtColor(lena, cv.COLOR_BGR2GRAY)

# 伽马变换
gammaLena = 1.0 * np.power(grayLena / 255.0, 1.5) * 255

gammaLena = gammaLena.astype(np.uint8)

gl2 = 1.0 * np.power(grayLena / 255.0, 0.5) * 255

plt.subplot(2, 3, 1)  # 设置子图分割方式与绘制位置
plt.imshow(grayLena, cmap='gray')  # 绘制子图
plt.axis('off')  # 关闭子图坐标轴
plt.title('lena')

plt.subplot(2, 3, 2)  # 设置子图分割方式与绘制位置
plt.imshow(gammaLena, cmap='gray')  # 绘制子图
plt.axis('off')  # 关闭子图坐标轴
plt.title('c = 1, gamma = 1.5')

# c = 1, gamma <1
# c >1 , g =1
# c <1 , g =1
# c >1 , g >1
plt.show()  # 显示绘图窗口

# 等待按键操作
cv.waitKey(0)
