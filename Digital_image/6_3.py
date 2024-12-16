import cv2 as cv  # 导入OpenCV 库
import numpy as np  # 导入Numpy 库
import matplotlib.pyplot as plt

# 读取图像,按照 BGR加载
img = cv.imread(r"E:\Digital_images\image\image\CH3\06001.tif")
imga = cv.imread(r"E:\Digital_images\image\image\CH3\06003a.tif")
imgb = cv.imread(r"E:\Digital_images\image\image\CH3\06003b.tif")
imgc = cv.imread(r"E:\Digital_images\image\image\CH3\06003c.tif")
# 缩放img 使图像大小一致
img = cv.resize(img, (imga.shape[1], imga.shape[0]))
# 转换色彩空间
grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

grayOpti = cv.cvtColor(imga, cv.COLOR_BGR2GRAY)  # 光学
grayXray = cv.cvtColor(imgb, cv.COLOR_BGR2GRAY)  # X射线
grayInfr = cv.cvtColor(imgc, cv.COLOR_BGR2GRAY)  # 红外

composite = np.zeros(img.shape, np.uint8)
# 叠加
composite[:, :, 0] = grayInfr
composite[:, :, 1] = grayXray
composite[:, :, 2] = grayOpti

grayImg1 = np.zeros(grayImg.shape, np.uint8)
grayImg1 = ((img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3).astype(np.uint8)

print(img[255, 100], grayImg[255, 100])
cv.imshow('img', grayImg)
cv.imshow('img1', grayImg1)
cv.imshow('composite', composite)
cv.waitKey(0)
cv.destroyAllWindows()
