import cv2 as cv  # 导入OpenCV 库
import numpy as np  # 导入Numpy 库
import matplotlib.pyplot as plt

# 读取图像,按照 BGR加载
img = cv.imread(r"E:\Digital_images\image\image\CH3\06001.tif")
# 转换色彩空间
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
HSIImg = cv.cvtColor(img, cv.COLOR_BGR2HSV)

imgR = img[:, :, 0]
imgG = img[:, :, 1]
imgB = img[:, :, 2]

img1 = np.zeros(img.shape, np.uint8)
img1[:, :, 0] = imgR  # R
img1[:, :, 1] = imgG  # G
img1[:, :, 2] = imgB  # B

cmyImg = 255 - img

cv.imshow('HSI', HSIImg)

# 循环显示
imgList = [img, grayImg, imgR, imgG, imgB, img1, cmyImg, HSIImg
           ]
imgTitle = ['1. Original', '2. gray', '3. R',
            '4. G', '5. B', '6. ', '7. CMY',
            '8. HSI '
            ]
imgStart = 331  # 子图起始位置，小于10幅子图时可用
imgNum = 8  # 子图数量，对应以上两个列表长度
for i in range(imgNum):
    plt.subplot(imgStart + i)
    plt.imshow(imgList[i], cmap='gray')
    plt.axis('off')
    plt.title(imgTitle[i])

plt.show()
