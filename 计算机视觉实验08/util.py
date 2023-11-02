import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy


# 高斯噪声
def normalize(mask, cut=True):
    if cut:
        return np.clip(mask, 0, 255) / 255.0
    return (mask - mask.min()) / (mask.max() - mask.min())


def add_gaussian_noise(img, mu=0, sigma=25):
    img = np.expand_dims(img, axis=-1) if img.ndim == 2 else img
    new_img = np.zeros(img.shape)
    for i in range(img.shape[2]):
        image = np.array(img[:, :, i], dtype=float)
        noise = np.random.normal(mu, sigma, image.shape)
        # print(np.mean(noise), np.std(noise))
        new_img[:, :, i] = normalize(image + noise) * 255
    if img.ndim == 2:
        new_img = np.squeeze(new_img, -1)
    return np.uint8(new_img)


# 瑞利噪声
def add_rayleigh_noise(img, mu=0, sigma=25):
    img = np.expand_dims(img, axis=-1) if img.ndim == 2 else img
    new_img = np.zeros(img.shape)
    for i in range(img.shape[2]):
        image = np.array(img[:, :, i], dtype=float)
        noise = np.random.rayleigh(scale=sigma, size=image.shape)
        noise = noise / np.std(noise) * sigma
        noise = noise - np.mean(noise) + mu
        # print(np.mean(noise), np.std(noise))
        new_img[:, :, i] = normalize(image + noise) * 255
        if img.ndim == 2:
            new_img = np.squeeze(new_img, -1)
    return np.uint8(new_img)


# 伽马噪声
def add_gamma_noise(img, mu=0, sigma=25):
    img = np.expand_dims(img, axis=-1) if img.ndim == 2 else img
    new_img = np.zeros(img.shape)
    for i in range(img.shape[2]):
        image = np.array(img[:, :, i], dtype=float)
        a = 2 * mu - np.sqrt(12 * sigma)
        b = 2 * mu + np.sqrt(12 * sigma)
        noise = np.random.uniform(a, b, image.shape)
        # showGrayHist(noise)
        # print(np.mean(noise), np.std(noise))
        # print(b*scale, b**0.5*scale)
        new_img[:, :, i] = normalize(image + noise) * 255
    if img.ndim == 2:
        new_img = np.squeeze(new_img, -1)
    return np.uint8(new_img)


# 椒盐噪声
def add_salt_pepper(img, ps=0.05, pp=0.05):
    img = np.expand_dims(img, axis=-1) if img.ndim == 2 else img
    new_img = np.zeros(img.shape)
    h, w = img.shape[:2]
    mask = np.random.choice((0, 0.5, 1), size=(h, w), p=[pp, (1 - pp - ps), ps])
    img_out = img
    img_out[mask == 1] = 255
    img_out[mask == 0] = 0
    new_img = img_out
    if img.ndim == 2:
        new_img = np.squeeze(new_img, -1)
    return np.uint8(new_img)


# 几何均值滤波
def GeometricMeanOperator(roi):
    roi = roi.astype(np.float64)
    p = np.prod(roi)
    return p ** (1 / (roi.shape[0] * roi.shape[1]))


def GeometricMeanAlogrithm(image):
    # 几何均值滤波
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            new_image[i - 1, j - 1] = GeometricMeanOperator(
                image[i - 1 : i + 2, j - 1 : j + 2]
            )
    new_image = (new_image - np.min(image)) * (255 / np.max(image))
    return new_image.astype(np.uint8)


def rgbGemoetricMean(image):
    r, g, b = cv2.split(image)
    r = GeometricMeanAlogrithm(r)
    g = GeometricMeanAlogrithm(g)
    b = GeometricMeanAlogrithm(b)
    return cv2.merge([r, g, b])


# 谐波均值滤波
def HarmonicMeanOperator(roi):
    roi = roi.astype(np.float64)
    if 0 in roi:
        roi = 0
    else:
        roi = scipy.stats.hmean(roi.reshape(-1))
    return roi


def HarmonicMeanAlogrithm(image):
    # 谐波均值滤波
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            new_image[i - 1, j - 1] = HarmonicMeanOperator(
                image[i - 1 : i + 2, j - 1 : j + 2]
            )
    new_image = (new_image - np.min(image)) * (255 / np.max(image))
    return new_image.astype(np.uint8)


def rgbHarmonicMean(image):
    r, g, b = cv2.split(image)
    r = HarmonicMeanAlogrithm(r)
    g = HarmonicMeanAlogrithm(g)
    b = HarmonicMeanAlogrithm(b)
    return cv2.merge([r, g, b])


# 逆谐波均值滤波
def Contra_harmonicMeanOperator(roi, q):
    roi = roi.astype(np.float64)
    return np.mean(roi ** (q + 1)) / np.mean((roi) ** (q))


def Contra_harmonicMeanAlogrithm(image, q):
    # 逆谐波均值滤波
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            new_image[i - 1, j - 1] = Contra_harmonicMeanOperator(
                image[i - 1 : i + 2, j - 1 : j + 2], q
            )
    new_image = (new_image - np.min(image)) * (255 / np.max(image))
    return new_image.astype(np.uint8)


def rgbContra_harmonicMean(image, q):
    r, g, b = cv2.split(image)
    r = Contra_harmonicMeanAlogrithm(r, q)
    g = Contra_harmonicMeanAlogrithm(g, q)
    b = Contra_harmonicMeanAlogrithm(b, q)
    return cv2.merge([r, g, b])


# 图像退化恢复
# 仿真运动模糊
def get_motion_dsf(image_sise, motion_angle, motion_dis):
    PSF = np.zeros(image_sise)  # 点扩散函数
    x_center = (image_sise[0] - 1) / 2
    y_center = (image_sise[1] - 1) / 2
    sin_val = np.sin(motion_angle * np.pi / 180)
    cos_val = np.cos(motion_angle * np.pi / 180)
    # 将对应角度上motion_dis个点置成1
    for i in range(motion_dis):
        x_offset = round(sin_val * i)
        y_offset = round(cos_val * i)
        PSF[int(x_center - x_offset), int(y_center + y_offset)] = 1
    return PSF / PSF.sum()  # 归一化


# 仿真湍流模糊
def cal_distance(pa, pb):  # 欧拉距离计算函数的定义
    return np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)


def get_turbulence_dsf(image_sise, k=0.1):
    center_point = tuple(map(lambda x: (x - 1) / 2, image_sise))  # 中心点
    pos_matrix = np.mgrid[0 : image_sise[0], 0 : image_sise[1]]  # 生成位置矩阵
    dis = cal_distance(pos_matrix, center_point)  # 计算距离矩阵
    # PSF_fft = dis
    PSF_fft = np.exp(-k * (dis ** (5 / 6)))  # 点扩散函数 fft
    PSF = np.fft.ifft2(PSF_fft)  # image FFT multiplly PSF FFT
    PSF = np.abs(np.fft.ifftshift(PSF))  # 中心化
    return PSF / PSF.sum()  # 归一化


# 图像退化恢复
# 对图片继续模糊
def make_blurred(img, PSF, eps):
    img = np.expand_dims(img, axis=-1) if img.ndim == 2 else img
    new_img = np.zeros(img.shape)
    for i in range(img.shape[2]):
        input_fft = np.fft.fft2(img[:, :, i])
        PSF_fft = np.fft.fft2(PSF) + eps
        blurred = np.fft.ifft2(input_fft * PSF_fft)
        blurred = np.abs(np.fft.ifftshift(blurred))
        new_img[:, :, i] = blurred
    if img.ndim == 2:
        new_img = np.squeeze(new_img, -1)
    return np.uint8(new_img)


def inverse_filter(img, PSF, eps):
    img = np.expand_dims(img, axis=-1) if img.ndim == 2 else img
    new_img = np.zeros(img.shape)
    for i in range(img.shape[2]):
        input_fft = np.fft.fft2(img[:, :, i])
        PSF_fft = np.fft.fft2(PSF) + eps
        result = np.fft.ifft2(input_fft / PSF_fft)
        result = np.abs(np.fft.ifftshift(result))
        new_img[:, :, i] = result
    if img.ndim == 2:
        new_img = np.squeeze(new_img, -1)
    return np.uint8(new_img)


def wiener_filter(img, PSF, eps, K=0.01):
    img = np.expand_dims(img, axis=-1) if img.ndim == 2 else img
    new_img = np.zeros(img.shape)
    for i in range(img.shape[2]):
        input_fft = np.fft.fft2(img[:, :, i])
        PSF_fft = np.fft.fft2(PSF) + eps
        PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
        result = np.fft.ifft2(input_fft * PSF_fft_1)
        result = np.abs(np.fft.ifftshift(result))
        new_img[:, :, i] = result
    if img.ndim == 2:
        new_img = np.squeeze(new_img, -1)
    return np.uint8(new_img)


def spectrum_show(PSF, title):
    PSF_fft = np.fft.fft2(PSF)
    PSF_fft_shift = np.fft.fftshift(PSF_fft)
    spectrum = 20 * np.log(np.abs(PSF_fft_shift))
    plt.imshow(spectrum, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


# 图像修补BSCB
def normalize(mask, cut=False):
    if cut:
        return np.clip(mask, 0, 255) / 255.0
    return (mask - mask.min()) / (mask.max() - mask.min())


# BSCB https://www.cnblogs.com/jgg54335/p/14561720.html


def BSCB_inpaint(
    pic_array,
    mask=None,
    epsilon=0.1,
    inpaint_iters=6,
    anidiffuse_iters=6,
    delta_ts=0.02,
    sensitivites=100,
    diffuse_coef=1,
):
    # BSCB算法
    pic_copy = pic_array.copy()
    # epsilon2 = epsilon ** 2
    epsilon2 = epsilon * epsilon
    pic_copy_ = pic_array.copy()
    for i in range(anidiffuse_iters):  # 执行各向异性扩散
        dx_dy = np.gradient(pic_copy)  # 计算梯度
        grad_norm = (
            dx_dy[0] ** 2 + dx_dy[1] ** 2 + epsilon2
        ) ** 0.5  # epsilon2是为了防止分母为0
        if diffuse_coef == 0:
            diffuse_coefs = np.exp(-grad_norm / sensitivites)
        else:
            diffuse_coefs = 1 / (1 + grad_norm / sensitivites)
        dxx = np.gradient(dx_dy[0], axis=0)
        dyy = np.gradient(dx_dy[1], axis=1)
        laplacian = dxx + dyy
        if not mask is None:
            diffuse_coefs = diffuse_coefs * mask
        pic_copy = pic_copy + diffuse_coefs * laplacian
    for i in range(inpaint_iters):  # 执行修补
        dx_dy = np.gradient(pic_copy)  # 计算梯度
        grad_norm = (dx_dy[0] ** 2 + dx_dy[1] ** 2 + epsilon2) ** 0.5
        dxx = np.gradient(dx_dy[0], axis=0)
        dyy = np.gradient(dx_dy[1], axis=1)
        laplacian = dxx + dyy
        dx_dy_ = np.gradient(laplacian)
        if not mask is None:
            delta_ts = delta_ts * mask
        delta_ts = delta_ts * (grad_norm > 0)
        pic_copy = (
            pic_copy
            - delta_ts * (-dx_dy[0] * dx_dy_[0] + dx_dy[1] * dx_dy_[1]) / grad_norm
        )
    # import time
    # timestamp = time.time()
    # print("time:", timestamp)
    # cv2.imwrite(str(timestamp) + ".jpg", img=np.uint8(pic_copy * 255))
    pic_new = pic_array.copy()
    pic_new[1:-1, 1:-1] = pic_copy[1:-1, 1:-1]  # 更新
    return pic_new


def gen_pic_with_mask(mask, origin_pic):
    origin_pic[mask == 1.0] = 128
    return origin_pic


# 图像修补TV
def normalize(mask, cut=False):
    if cut:
        return np.clip(mask, 0, 255) / 255.0
    return (mask - mask.min()) / (mask.max() - mask.min())


# TV https://www.cnblogs.com/hxjbc/p/6675901.html


def tv_inpaint(pic_array, mask=None, epsilon=0.1, dt=0.1, lambda_=0.1, withCCD=True):
    # tv算法
    pic_copy = pic_array.copy()
    epsilon2 = epsilon * epsilon
    # 求梯度
    dx_dy = np.gradient(pic_copy)
    dx_dy = dx_dy / (dx_dy[0] ** 2 + dx_dy[1] ** 2 + epsilon2) ** 0.5
    # 求散度 divergence(zx, zy) = zx_x + zy_y
    dxx = np.gradient(dx_dy[0], axis=0)
    dyy = np.gradient(dx_dy[1], axis=1)
    div = dxx + dyy
    if withCCD:  # 是否曲率驱动
        k = np.abs(div)
        k = k / np.max(k) / dt
        k = 0.3 + 0.7 * k**0.1
        div = div * k
    if not mask is None:
        div = div * mask
    # 迭代求解
    pic_copy = pic_copy + dt * div - lambda_ * (pic_copy - pic_array)
    pic_new = pic_array.copy()
    pic_new[1:-1, 1:-1] = pic_copy[1:-1, 1:-1]  # 更新
    return pic_new


def gen_pic_with_mask(mask, origin_pic):
    origin_pic[mask == 1.0] = 128
    return origin_pic
