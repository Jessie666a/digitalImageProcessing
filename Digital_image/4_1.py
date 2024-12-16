import cv2 as cv  # 导入OpenCV 库
import numpy as np  # 导入Numpy 库
import matplotlib.pyplot as plt

# from matplotlib import pyplot as plt

# 读取图像
img = cv.imread(r"E:\Digital_images\image\image\CH3\04001.tif")

# 转换色彩空间
grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


print(grayImg.shape)
# 傅里叶变换 方式一
# 正变换
# dft = cv.dft( grayImg.astype(np.float32) ,flags = cv.DFT_COMPLEX_OUTPUT) # 傅里叶变换
dft = np.fft.fft2(grayImg)
dftShift = np.fft.fftshift(dft)  # 中心化
print(dft.shape)
# 反变换
idftShift = np.fft.ifftshift(dftShift)  # 反中心化
# idft = cv.dft( idftShift ,flags = cv.DFT_INVERSE) # 傅里叶逆变换
idft = np.fft.ifft2(idftShift)
rebuild = cv.magnitude(idft[:, :, 0], idft[:, :, 1])  # 重建图像
rebuild2 = cv.normalize(rebuild, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)  # 缩放结果
# print(min(grayImg.ravel()),max(grayImg.ravel()))
# print(min(rebuild2.ravel()),max(rebuild2.ravel()))

# 频谱
dftAmp = cv.magnitude(dft[:, :, 0], dft[:, :, 1])  # 频谱
AmpShift = np.fft.fftshift(dftAmp)  # 中心化频谱
logAmpShift = np.log(1 + AmpShift)  # 对数变换
phase = np.arctan2(dft[:, :, 1], dft[:, :, 0])  # 相位（弧度制）
phi = phase / np.pi * 180  # 相位角
#
print('原图像 min~max：', min(grayImg.ravel()), max(grayImg.ravel()))
print('DFT 频谱 min~max：', min(dftAmp.ravel()), max(dftAmp.ravel()))
print('DFT 中心化频谱 min~max：', min(AmpShift.ravel()), max(AmpShift.ravel()))
print('DFT 相位 min~max：', min(phi.ravel()), max(phi.ravel()))

# 循环显示
imgList = [grayImg, dftAmp, AmpShift, logAmpShift, phi, rebuild2]
imgTitle = ['1. Original', '2.dft Amp', '3. center Amp', '4.log Amp', '5. phase(Phi)', '6. dft']
imgStart = 231  # 子图起始位置，小于10幅子图时可用
imgNum = 6  # 子图数量，对应以上两个列表长度
for i in range(imgNum):
    plt.subplot(imgStart + i)
    plt.imshow(imgList[i], cmap='gray')
    plt.axis('off')
    plt.title(imgTitle[i])

plt.show()
