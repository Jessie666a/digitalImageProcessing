import cv2 as cv  # 导入OpenCV 库
import numpy as np  # 导入Numpy 库
import matplotlib.pyplot as plt


# from matplotlib import pyplot as plt

# 理想带阻滤波器
def myIBRF(shape, C0, W):
    height, width = shape
    # 测度网格矩阵
    u, v = np.mgrid[-width // 2:(width + 1) // 2, -height // 2:(height + 1) // 2]
    # 计算频率点的测度距离
    D = np.sqrt(u ** 2 + v ** 2)
    # 创建初始化滤波器矩阵
    H = np.ones((width, height, 2), dtype=np.float32)
    # 更新滤波矩阵
    H[(D <= C0 + W / 2) & (D >= C0 - W / 2)] = 0
    return H


# 高斯带阻滤波器
def myGBRF(shape, C0, W):
    height, width = shape
    # 测度网格矩阵
    u, v = np.mgrid[-width // 2:(width + 1) // 2, -height // 2:(height + 1) // 2]
    # 计算频率点的测度距离
    D = np.sqrt(u ** 2 + v ** 2)
    # 创建初始化滤波器矩阵
    H = np.ones((width, height), dtype=np.float32)
    # 更新滤波矩阵
    H = 1 - np.exp(-((D ** 2 - C0 ** 2) / (D * W + 1e-08)) ** 2)
    H_complpex = np.zeros((width, height, 2), dtype=np.float32)
    H_complpex[:, :, 0] = H
    return H_complpex


# 巴特沃斯带阻滤波器
def myBBRF(shape, C0, W, n):
    height, width = shape
    # 测度网格矩阵
    u, v = np.mgrid[-width // 2:(width + 1) // 2, -height // 2:(height + 1) // 2]
    # 计算频率点的测度距离
    D = np.sqrt(u ** 2 + v ** 2)
    # 创建初始化滤波器矩阵
    H = np.ones((width, height), dtype=np.float32)
    # 更新滤波矩阵
    H = 1 / (1 + (D * W / (D ** 2 - C0 ** 2 + 1e-08)) ** 2 * n)
    H_complpex = np.zeros((width, height, 2), dtype=np.float32)
    H_complpex[:, :, 0] = H
    return H_complpex


# 傅里叶变换函数
def myDFTandBPF(myImg, flag, C0, W, n=0):
    # 1. 数据格式转换
    floatImg = myImg.astype(np.float32)
    # 2. 傅里叶变换
    dft = cv.dft(floatImg, flags=cv.DFT_COMPLEX_OUTPUT)
    # 3. 中心化
    dftShift = np.fft.fftshift(dft)
    # 4. 带阻滤波器
    if flag == 'IBRF' or flag == 0:  # 理想带阻
        H = myIBRF(myImg.shape, C0, W)
    elif flag == 'GBRF' or flag == 1:  # 高斯带阻
        H = myGBRF(myImg.shape, C0, W)
    elif flag == 'BBRF' or flag == 2:  # 巴特沃斯带阻
        H = myBBRF(myImg.shape, C0, W, n)

    BRF = H * dftShift

    # 5. 去中心化
    iShift = np.fft.ifftshift(BRF)
    # 6. 傅里叶逆变换
    idft = cv.idft(iShift)
    # 7. 重建图像
    idftAmp = cv.magnitude(idft[:, :, 0], idft[:, :, 1])
    rebuild = cv.normalize(idftAmp, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    # 频谱结果


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
img = cv.imread(r"D:\Code\Python\class02\CH3\04004.tif")
# 转换色彩空间
grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

BRFimg = myDFTandBPF(grayImg, 1, 100, 50)

# 循环显示
imgList = [grayImg, BRFimg[0], BRFimg[4][:, :, 0]]
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
ax.plot_surface(u, v, BRFimg[4][:, :, 0], cmap='viridis')
plt.axis('off')

plt.show()
