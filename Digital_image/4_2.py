import cv2 as cv  # 导入OpenCV 库
import numpy as np  # 导入Numpy 库
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
# 傅里叶变换函数
def myDFT(myImg):
    # 1. 数据格式转换
    floatImg = myImg.astype(np.float32)

    # 2. 傅里叶变换
    dft = cv.dft(floatImg, flags=cv.DFT_COMPLEX_OUTPUT)

    # 3. 中心化
    dftShift = np.fft.fftshift(dft)
    # 4. 去中心化
    iShift = np.fft.ifftshift(dftShift)


    # 5. 傅里叶逆变换
    idft = cv.idft(iShift)


    # 6. 重建图像
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
    return [rebuild, dftAmpShift, dftLogShift, dftPhi]



# 读取图像
img = cv.imread(r"E:\Digital_images\image\image\CH3\04001.tif")
# 转换色彩空间
grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)



# 1. 获取图像的宽高
height, width = grayImg.shape
# 2. 计算图像中心位置
center = (width / 2, height / 2)
# 3. 计算旋转的仿射矩阵
dst = cv.getRotationMatrix2D(center, -45, 1)
# 4. 变换得到旋转结果图像
rotationImg = cv.warpAffine(grayImg, dst, (width, height))



DFTimg = myDFT(grayImg)  # 原图像做傅里叶变换
DFTro = myDFT(rotationImg)  # 旋转图像做傅里叶变换
# 读取图像
img = cv.imread(r"E:\Digital_images\image\image\CH3\04002.tif")
# 转换色彩空间
grayImg1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
DFTmove = myDFT(grayImg1)  # 平移图像做傅里叶变换

# 循环显示
imgList = [grayImg, DFTimg[2], DFTimg[3],
           rotationImg, DFTro[2], DFTro[3],
           grayImg1, DFTmove[2], DFTmove[3]]
imgTitle = ['1. Original', '2. log spectrogram', '3. phase ',
            '4. Rotation', '5. log spectrogram', '6. phase',
            '7. Move    ', '8. log spectrogram', '9. phase']
imgStart = 331  # 子图起始位置，小于10幅子图时可用
imgNum = 9  # 子图数量，对应以上两个列表长度
for i in range(imgNum):
    plt.subplot(imgStart + i)
    plt.imshow(imgList[i], cmap='gray')
    plt.axis('off')
    plt.title(imgTitle[i])

plt.show()
