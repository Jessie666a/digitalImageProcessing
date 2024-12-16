import cv2 as cv  # 导入OpenCV 库
import numpy as np  # 导入Numpy 库

# 读取图像
lena = cv.imread(r"E:/Digital_images/image/image/lena1.jpg")
# 转换色彩空间
grayLena = cv.cvtColor(lena, cv.COLOR_BGR2GRAY)

# 系统自动分配一片内存空间
img = np.empty(grayLena.shape, dtype=np.uint8)
print(img)

cv.imshow('img', img)

# 等待按键操作
cv.waitKey(0)
