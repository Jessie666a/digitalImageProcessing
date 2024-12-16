import cv2 as cv  # 导入OpenCV 库
import numpy as np  # 导入Numpy 库
import matplotlib.pyplot as plt

# from matplotlib import pyplot as plt

# 读取图像
lena = cv.imread(r"E:\Digital_images\image\image\lena.jpg")
# 转换色彩空间
grayLena = cv.cvtColor(lena, cv.COLOR_BGR2GRAY)

# 均衡化
equlLena = cv.equalizeHist(grayLena)

plt.subplot(221)
plt.imshow(grayLena, cmap='gray')

plt.subplot(222)
plt.hist(grayLena.ravel(), 256, (0, 256))  # 不确定此处是否修改正确，待后期验证

plt.subplot(223)
plt.imshow(equlLena, cmap='gray')

plt.subplot(224)
plt.hist(equlLena.ravel(), 256, (0, 256))

plt.show()
cv.waitKey(0)
