import cv2 as cv  # 导入OpenCV 库
import numpy as np  # 导入Numpy 库
import matplotlib.pyplot as plt


# from matplotlib import pyplot as plt

# 傅里叶变换函数
def myDFTandILPF(myImg):
    # 1. 数据格式转换
    floatImg = myImg.astype(np.float32)
    # 2. 傅里叶变换
    dft = cv.dft(floatImg, flags=cv.DFT_COMPLEX_OUTPUT)
    # 3. 中心化
    dftShift = np.fft.fftshift(dft)
    # 4. 理想低通滤波器 ILPF
    # 设置截止频率
    D0 = 50
    # 创建滤波器函数H
    height, width = dftShift.shape[:2]
    H = np.zeros((width, height, 2), dtype=np.float32)
    for u in range(width):
        for v in range(height):
            D = np.sqrt((u - width // 2) ** 2 + (v - height // 2) ** 2)  # 计算距离
            if D <= D0:
                H[u, v, 0] = 1
                H[u, v, 1] = 1
    # 低通滤波
    ILPF = H * dftShift
    # 5. 去中心化
    iShift = np.fft.ifftshift(ILPF)
    # 6. 傅里叶逆变换
    idft = cv.idft(iShift)
    # 7. 重建图像
    idftAmp = cv.magnitude(idft[:, :, 0], idft[:, :, 1])
    rebuild = cv.normalize(idftAmp, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    ### 频谱结果
    #
    dftAmp = cv.magnitude(dft[:, :, 0], dft[:, :, 1])
    ampLog = np.log1p(dftAmp)
    # 中心化频谱
    dftAmpShift = cv.magnitude(dftShift[:, :, 0], dftShift[:, :, 1])
    # 频谱对数结果
    dftLogShift = np.log1p(dftAmpShift)
    # 相位谱
    phase = np.arctan2(dft[:, :, 1], dft[:, :, 0])  # 弧度制
    dftPhi = phase / np.pi * 180  # 角度制
    # 返回重建图像、中心化频谱、频谱对数结果、相位谱
    return [rebuild, dftAmpShift, dftLogShift, dftPhi, H]


# 读取图像
img = cv.imread(r"E:\Digital_images\image\image\CH3\04001.tif")
# 转换色彩空间
grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ILPFimg = myDFTandILPF(grayImg)

# 循环显示
imgList = [grayImg, ILPFimg[0], ILPFimg[4][:, :, 0]]
imgTitle = ['1. Original', '2. rebuild', '3. phase ',
            '4. Rotation', '5. log spectrogram', '6. phase',
            '7. Move    ', '8. log spectrogram', '9. phase']
imgStart = 141  # 子图起始位置，小于10幅子图时可用
imgNum = 3  # 子图数量，对应以上两个列表长度
for i in range(imgNum):
    plt.subplot(imgStart + i)
    plt.imshow(imgList[i], cmap='gray')
    plt.axis('off')
    plt.title(imgTitle[i])

ax = plt.subplot(144, projection='3d')
u, v = np.mgrid[-1:1:2.0 / grayImg.shape[1], -1:1:2.0 / grayImg.shape[0]]
# ax.plot_wireframe( u,v, ILPF ,rstride=100, cstride=10, linewidth=0.2)
ax.plot_surface(u, v, ILPFimg[4][:, :, 0], cmap='viridis')
plt.axis('off')

plt.show()
