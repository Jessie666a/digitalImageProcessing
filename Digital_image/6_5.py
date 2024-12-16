import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv.imread(r"D:\Code\Python\class02\CH3\06004.tif")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 伽马变换
gamaimg = np.power(img, 0.5)
# 归一化处理
gamaimg = gamaimg - np.min(gamaimg)  # 左侧范围调整为0
gamaimg = gamaimg / np.max(gamaimg)  # 右侧范围调整为1

# 变换到CMY 模型
CMY = 1 - gamaimg
# 拆分通道
c, m, y = cv.split(CMY)
# 对m 通道进行调节
m = np.power(m, 1.2)
m = m - np.min(m)  # 左侧范围调整为0
m = m / np.max(m)  # 右侧范围调整为1

# 三通道合成
out = cv.merge([c, m, y])
# 切换到RGB 模型
out = 1 - out
# 调整范围用于显示
out = np.uint8(255 * out)
gamaimg = np.uint8(255 * gamaimg)

plt.subplot(131), plt.imshow(img), plt.title('Original'), plt.axis('off')
plt.subplot(132), plt.imshow(gamaimg), plt.title('gama'), plt.axis('off')
plt.subplot(133), plt.imshow(out), plt.title('out'), plt.axis('off')

plt.show()
