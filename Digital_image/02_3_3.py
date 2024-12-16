import cv2 as cv
import numpy as np

panda = cv.imread(r"E:/Digital_images/image/image/pandamin.jpeg")
print(panda[0, 0])
print(panda[0, 0, 0])
print(panda[0, 0, 1])
print(panda[0, 0, 2])
# BGR
cv.imshow('B', panda[:, :, 0])
cv.imshow('G', panda[:, :, 1])
cv.imshow('R', panda[:, :, 2])

cv.waitKey(0)
cv.destroyAllWindows()