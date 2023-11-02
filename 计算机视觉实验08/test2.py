from util import *


# 【2】任选图片，用退化模型进行退化模拟，并通过维纳滤波进行复原，需测试噪声的影响。
# 退化模拟
image = cv2.imread('./imgs/cat.jpg')

# 显示原图像
plt.figure(1, figsize=(6, 6))
plt.title('Original Image'), plt.imshow(image[..., [2, 1, 0]], 'gray')
plt.xticks([]), plt.yticks([])


# 进行模糊处理
PSF = get_motion_dsf(image.shape[:2], -50, 100)
PSF = get_turbulence_dsf(image.shape[:2])
spectrum_show(PSF, title='PSF Image')  # 模糊PSF与谱
blurred = make_blurred(image, PSF, 1e-3)
plt.figure(2, figsize=(8, 8))
plt.subplot(231), plt.imshow(
    blurred[..., [2, 1, 0]], 'gray'), plt.title('blurred')
plt.xticks([]), plt.yticks([])

# 逆滤波
result = inverse_filter(blurred, PSF, 1e-3)
plt.subplot(232), plt.imshow(
    result[..., [2, 1, 0]], 'gray'), plt.title('inverse deblurred')
plt.xticks([]), plt.yticks([])

# 维纳滤波
result = wiener_filter(blurred, PSF, 1e-3)
plt.subplot(233), plt.imshow(result[..., [2, 1, 0]], 'gray'), plt.title(
    'wiener deblurred(K=0.01)')
plt.xticks([]), plt.yticks([])

# 添加噪声,standard_normal产生随机的函数
blurred_noisy = np.uint8(blurred + 0.1*blurred.std()
                         * np.random.standard_normal(blurred.shape))

# 显示添加噪声且模糊的图像
plt.subplot(234), plt.imshow(
    blurred_noisy[..., [2, 1, 0]], 'gray'), plt.title('blurred and noisy')

# 对添加噪声的图像进行逆滤波
result = inverse_filter(blurred_noisy, PSF, 0.1+1e-3)
plt.subplot(235), plt.imshow(
    result[..., [2, 1, 0]], 'gray'), plt.title('inverse deblurred')
plt.xticks([]), plt.yticks([])

# 对添加噪声的图像进行维纳滤波
result = wiener_filter(blurred_noisy, PSF, 0.1+1e-3)
plt.subplot(236), plt.imshow(result[..., [2, 1, 0]], 'gray'), plt.title(
    'wiener deblurred(k=0.01)')
plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.savefig('out.jpg', format='jpg', bbox_inches='tight', dpi=96)
plt.show()
