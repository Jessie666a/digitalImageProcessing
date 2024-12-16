import cv2 as cv  # 导入 opencv 库

panda = cv.imread(r"E:/Digital_images/image/image/pandamin.jpeg")  # 读取彩色图像
lena = cv.imread(r"E:/Digital_images/image/image/lena.jpg")  # 读取彩色图像

panda = cv.resize(panda, (lena.shape[1], lena.shape[0]))  # 调整图像大小到一致

print(lena.shape, panda.shape)
print(panda[0, 0])
print(lena[0, 0])
pandaLena = cv.add(panda, lena)  # opencv 加法
print(pandaLena[0, 0])
cv.imshow('pandaLena', pandaLena)

out = cv.addWeighted(panda, 0.4, lena, 0.4, 0)  # opencv 权重加法
print(out[0, 0])
cv.imshow('out', out)

cv.waitKey(0)
