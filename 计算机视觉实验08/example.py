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


# ----------------------------------------------

img = cv2.imread('dog_defiled.jpg')
# 图片二值化处理，把[0, 0, 200]~[70, 70, 255] 以外的颜色变成0
thresh = cv2.inRange(img, np.array([0, 0, 200]), np.array([70, 70, 255]))
# 创建形状和尺寸的结构元素
kernel = np.ones((3, 3), np.uint8)
# 扩张待修复区域
mask = cv2.dilate(thresh, kernel, iterations=1)/255.0
specturm_show(mask)

pic = gen_pic_with_mask(mask, img)

epsilon = 0.1
inpaint_iters = 6
anidiffuse_iters = 6
delta_ts = 0.2
sensitivites = 100
diffuse_coef = 1

epochs = 201
pic = (pic/255.0).astype(np.float)
pic_copy = np.zeros(pic.shape)
for epoch in range(epochs):
    # 每epochs次显示一次数据，保存一次数据
    if epoch % 40 == 0:
        print('epoch,当前的循环次数：', epoch, np.abs(pic-pic_copy).max())
        cv2.imwrite('dog_filed'+str(epoch)+'.jpg', img=np.uint8(pic*255))
    pic_copy = pic.copy()
    if epoch < epoch-1:
        for i in range(3):
            pic[:, :, i] = BSCB_inpaint(pic_copy[:, :, i], mask=mask,
                                        epsilon=epsilon, inpaint_iters=inpaint_iters, anidiffuse_iters=anidiffuse_iters, delta_ts=delta_ts,
                                        sensitivites=sensitivites, diffuse_coef=diffuse_coef)
pic = np.uint8(pic*255)
specturm_show(pic)

# ----------------------------------------------


img = cv2.imread('dog_defiled.jpg')
# 图片二值化处理，把[240, 240, 240]~[255, 255, 255] 以外的颜色变成0
thresh = cv2.inRange(img, np.array([0, 0, 200]), np.array([70, 70, 255]))
# 创建形状和尺寸的结构元素
kernel = np.ones((3, 3), np.uint8)
# 扩张待修复区域
mask = cv2.dilate(thresh, kernel, iterations=1)/255.0
specturm_show(mask)

pic = gen_pic_with_mask(mask, img)

epsilon = 0.1
dt = 0.1
lambda_ = 0.1
epochs = 601
pic = (pic/255.0).astype(np.float)
pic_copy = np.zeros(pic.shape)

for epoch in range(epochs):
    # 每epochs次显示一次数据，保存一次数据
    if epoch % 100 == 0:
        print('epoch,当前的循环次数：', epoch, np.abs(pic-pic_copy).max())
        cv2.imwrite('dog_filed'+str(epoch)+'.jpg', img=np.uint8(pic*255))
    pic_copy = pic.copy()
    if epoch < epoch-1:
        for i in range(3):
            pic[:, :, i] = tv_inpaint(pic_copy[:, :, i], mask=mask,
                                      epsilon=epsilon, dt=dt, lambda_=lambda_)
pic = np.uint8(pic*255)
specturm_show(pic)


# 图像修补NS、FMM
img = cv2.imread('dog_defiled.jpg')
# 图片二值化处理，把[240, 240, 240]~[255, 255, 255] 以外的颜色变成0
thresh = cv2.inRange(img, np.array([0, 0, 200]), np.array([70, 70, 255]))
# 创建形状和尺寸的结构元素
kernel = np.ones((3, 3), np.uint8)
# 扩张待修复区域
mask = cv2.dilate(thresh, kernel, iterations=1)
specturm_show(mask)

out = cv2.inpaint(img, mask, inpaintRadius=-1, flags=cv2.INPAINT_TELEA)

specturm_show(out)
cv2.imshow('dog_mask.jpg', mask)
cv2.imwrite('dog_filed.jpg', out)

# 图像修补PatchMatch
