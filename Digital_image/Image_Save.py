import cv2
image = cv2.imread(r"E:/Digital_images/image/image/pandamin.jpeg")
cv2.imwrite("E:/Digital_images/new.jpeg", image)


# 此处留意，保存图片需要加上图片保存格式后缀名.jpeg