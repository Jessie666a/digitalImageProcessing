import cv2 as cv  # 导入OpenCV 库
import numpy as np  # 导入Numpy 库

# 读取图像
lena = cv.imread(r"E:/Digital_images/image/image/lena.jpg")
# 转换色彩空间
grayLena = cv.cvtColor(lena, cv.COLOR_BGR2GRAY)

# 对数变换
logLena = 1.0 * np.log1p(grayLena)
print(max(logLena.ravel()))


# 缩放对数运算结果，将原范围变化到[0,255]
logLena = (logLena * 255 / max(logLena.ravel())).astype(np.uint8)
print(logLena)
cv.imshow('logLena', logLena)

# 等待按键操作
cv.waitKey(0)
