import cv2 as cv  # 导入 opencv 库

img = cv.imread(r"E:/Digital_images/image/image/06003.jpg")  # 读取彩色图像

print(type(img))  # img 为对应的像素值，但是是numpy 中的数组类型
print(img.shape)  # 图像信息（高，宽，通道数）

Nearest = cv.resize(img, (1000, 1000), interpolation=cv.INTER_NEAREST)  # 图像放大，最临近插值法
Linear = cv.resize(img, (1000, 1000), interpolation=cv.INTER_LINEAR)  # 图像放大，双线性插值法
Area = cv.resize(img, (1000, 1000), interpolation=cv.INTER_AREA)  # 图像放大，区域插值（一般用在缩小）
Cubic = cv.resize(img, (1000, 1000), interpolation=cv.INTER_CUBIC)  # 图像放大，双三次插值
Lanczos4 = cv.resize(img, (1000, 1000), interpolation=cv.INTER_LANCZOS4)  # 图像放大，Lanczos插值

cv.imwrite(r'E:/Digital_images/image/pandaCubic.jpg', Cubic)  # 保存图像数据

cv.imshow('img', img)  # 显示图像，指定窗口名为 img
cv.imshow('Nearest', Nearest)  # 显示图像，指定窗口名为 Nearest
cv.imshow('Linear', Linear)  # 显示图像，指定窗口名为 Linear
cv.imshow('Area', Area)  # 显示图像，指定窗口名为 Area
cv.imshow('Cubic', Cubic)  # 显示图像，指定窗口名为 Cubic
cv.imshow('Lanczos4', Lanczos4)  # 显示图像，指定窗口名为 Lanczos4

cv.waitKey(0)  # 等待按键操作
