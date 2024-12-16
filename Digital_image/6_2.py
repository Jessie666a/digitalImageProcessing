import cv2 as cv  # 导入OpenCV 库

ColormapTypes = (
    cv.COLORMAP_AUTUMN,  # 0
    cv.COLORMAP_BONE,  # 1
    cv.COLORMAP_JET,  # 2
    cv.COLORMAP_WINTER,  # 3
    cv.COLORMAP_RAINBOW,  # 4
    cv.COLORMAP_OCEAN,  # 5
    cv.COLORMAP_SUMMER,  # 6
    cv.COLORMAP_SPRING,  # 7
    cv.COLORMAP_COOL,  # 8
    cv.COLORMAP_HSV,  # 9
    cv.COLORMAP_PINK,  # 10
    cv.COLORMAP_HOT,  # 11
    cv.COLORMAP_PARULA,  # 12
    cv.COLORMAP_MAGMA,  # 13
    cv.COLORMAP_INFERNO,  # 14
    cv.COLORMAP_PLASMA,  # 15
    cv.COLORMAP_VIRIDIS,  # 16
    cv.COLORMAP_CIVIDIS,  # 17
    cv.COLORMAP_TWILIGHT,  # 18
    cv.COLORMAP_TWILIGHT_SHIFTED,  # 19
    cv.COLORMAP_TURBO,  # 20
    cv.COLORMAP_DEEPGREEN  # 21
)

# 读取图像,按照 BGR加载
img = cv.imread(r"E:\Digital_images\image\image\CH3\06001.tif")
# 转换色彩空间


grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
index = 0
while True:
    dst = cv.applyColorMap(grayImg, index)
    index = index + 1
    cv.imshow('apply color', dst)
    if index > len(ColormapTypes) - 1:
        break
    key = cv.waitKey(1000)
    if key == ord('q'):
        break

cv.destroyAllWindows()

