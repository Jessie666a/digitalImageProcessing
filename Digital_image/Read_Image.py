import cv2
# image = cv2.imread("E:/Digital_images/image/image/06003.jpg")
# print(image)
def main():

    img = cv2.imread("E:/Digital_images/image/image/06003.jpg", 0)
# 显示出的是一张灰度图，原因是
    # 在cv2.imread()
    # 函数中，第二个参数用于指定图像的读取模式。当第二个参数为0时，它会以灰度模式（单通道）读取图像。
    # 这就是为什么你传入彩色图片，但显示的是黑白图的原因。
    # 在彩色图像中，通常有三个通道（RGB，即红、绿、蓝），而灰度模式下，图像被转换为单通道，
    # 每个像素点只用一个亮度值来表示，从而呈现出黑白效果。
    cv2.imshow("img", img)
    cv2.waitKey(0)
# cv2.waitKey() 主要用于等待键盘事件，它的参数表示等待的时间（单位是毫秒）。
# 当传入参数为 0 时，意味着会无限期地等待用户按下键盘上的某个按键，
# 只有当检测到按键按下这个动作后，程序才会继续往下执行

# 注释要空两行


main()
