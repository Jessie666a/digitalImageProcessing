import cv2 as cv  # 导入OpenCV 库
import numpy as np  # 导入Numpy 库
import matplotlib.pyplot as plt
import math


# 仿真运动模糊
def motion_process(image_size, motion_angle):
    PSF = np.zeros(image_size)
    center_position = (image_size[0] - 1) / 2
    slope_tan = math.tan(motion_angle * math.pi / 180)
    slope_cot = 1 / slope_tan
    if slope_tan <= 1:
        for i in range(15):
            offset = round(i * slope_tan)  # ((center_position-i)*slope_tan)
            PSF[int(center_position + offset), int(center_position - offset)] = 1
        return PSF / PSF.sum()  # 对点扩散函数进行归一化亮度
    else:
        for i in range(15):
            offset = round(i * slope_cot)
            PSF[int(center_position - offset), int(center_position + offset)] = 1
        return PSF / PSF.sum()


# 对图片进行运动模糊
def make_blurred(input, PSF, eps):
    input_fft = np.fft.fft2(input)  # 进行二维数组的傅里叶变换
    PSF_fft = np.fft.fft2(PSF) + eps
    blurred = np.fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(np.fft.fftshift(blurred))
    return blurred


# 逆滤波
def inverse(input, PSF, eps):
    # 退化图像做傅里叶变换
    input_fft = np.fft.fft2(input)
    # 退化模型做傅里叶变换
    PSF_fft = np.fft.fft2(PSF) + eps
    # 复原运算，得到复原图像
    output = np.fft.ifft2(input_fft / PSF_fft)
    output = np.abs(np.fft.fftshift(output))
    return output


# 维纳滤波
def wiener(input, PSF, eps, k=0.01):
    # 退化图像做傅里叶变换
    input_fft = np.fft.fft2(input)
    # 退化模型做傅里叶变换
    PSF_fft = np.fft.fft2(PSF) + eps
    PSF_fft1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + k)
    # 复原运算，得到复原图像
    output = np.fft.ifft2(input_fft * PSF_fft1)
    output = np.abs(np.fft.fftshift(output))
    return output


# 读取图像
img = cv.imread(r"D:\Code\Python\class02\CH3\04006.tif")
# 转换色彩空间
grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 仿真运动模糊
PSF = motion_process(grayImg.shape, 60)
# （退化）给图像添加运动模糊
blurred_img = make_blurred(grayImg, PSF, 1e-03)
# 添加噪声,standard_normal产生随机的函数
blurred_noisy = blurred_img + 0.1 * blurred_img.std() * np.random.standard_normal(blurred_img.shape)
# 逆滤波
inverse_img = inverse(blurred_img, PSF, 1e-03)
ninverse_img = inverse(blurred_noisy, PSF, 0.3 + 1e-03)
# 维纳滤波
wiener_img = wiener(blurred_img, PSF, 1e-03)
nwiener_img = wiener(blurred_noisy, PSF, 1e-03)

# 循环显示
imgList = [grayImg, blurred_img, blurred_noisy,
           inverse_img, ninverse_img, wiener_img,
           nwiener_img
           ]
imgTitle = ['1. Original', '2. blurred_img', '3. blurred_noisy',
            '4. inverse_img', '5. ninverse_img', '6. wiener_img',
            '7. nwiener_img'
            ]
imgStart = 241  # 子图起始位置，小于10幅子图时可用
imgNum = 7  # 子图数量，对应以上两个列表长度
for i in range(imgNum):
    plt.subplot(imgStart + i)
    plt.imshow(imgList[i], cmap='gray')
    plt.axis('off')
    plt.title(imgTitle[i])
plt.show()
