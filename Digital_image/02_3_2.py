import cv2 as cv
import numpy as np

# 加载图像
lena = cv.imread(r"E:\Digital_images\image\image\lena.jpg")
lena1 = cv.imread(r"E:\Digital_images\image\image\lena1.jpg")

img1 = cv.subtract(lena1, lena)
img2 = lena1 - lena

cv.imshow('img1', img1)
cv.imshow('img2', img2)

# add img1 img2  img3
# subtract img2

lena = cv.imread(r"E:\Digital_images\image\image\lena.jpg")
panda = cv.imread(r"E:\Digital_images\image\image\pandamin.jpeg")
panda = cv.resize(panda, (lena.shape[1], lena.shape[0]))

pandalena = cv.add(lena, panda)
img3 = cv.subtract(pandalena, lena)
img4 = pandalena - lena
cv.imshow('img3', img3)
cv.imshow('img4', img4)

img5 = np.zeros((panda.shape[1], panda.shape[0], panda.shape[2]))
img6 = np.ones((panda.shape[1], panda.shape[0], panda.shape[2])) * 255
print(img6)
cv.imshow('img5', img5)
cv.imshow('img6', img6)

cv.waitKey(0)
