from util import *


# def specturm_show(img: cv2.UMat):
#     # 转换为灰度图像
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # 计算傅里叶变换
#     f = np.fft.fft2(gray)
#     fshift = np.fft.fftshift(f)

#     # 计算频谱
#     magnitude_spectrum = 20*np.log(np.abs(fshift))

#     # 显示频谱
#     plt.imshow(magnitude_spectrum, cmap='gray')
#     plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#     plt.show()


# 【3】选合适的图片，进行破损图像修补的实验。
img = cv2.imread("./imgs/dog.jpg")
# 图片二值化处理，把[0, 0, 200]~[70, 70, 255] 以外的颜色变成0
# thresh = cv2.inRange(img, np.array([0, 0, 200]), np.array([70, 70, 255]))
thresh = cv2.inRange(img, np.array([110, 50, 50]), np.array([130, 255, 255]))
# 创建形状和尺寸的结构元素
kernel = np.ones((3, 3), np.uint8)
# 扩张待修复区域
mask = cv2.dilate(thresh, kernel, iterations=1) / 255.0

# specturm_show(mask)
# spectrum_show(mask, "title")

pic = gen_pic_with_mask(mask, img)
cv2.imwrite("destroyedDog.jpg", img=pic)


# cv2.imshow("original", img)
# cv2.imshow("mask", mask)
# cv2.imshow("pic", pic)

epsilon = 0.1
inpaint_iters = 6
anidiffuse_iters = 6
delta_ts = 0.2
sensitivites = 100
diffuse_coef = 1

epochs = 201
# pic = (pic/255.0).astype(np.float)
pic = (pic / 255.0).astype(float)
pic_copy = np.zeros(pic.shape)
for epoch in range(epochs):
    # 每epochs次显示一次数据，保存一次数据
    if epoch % 40 == 0:
        print("epoch,当前的循环次数：", epoch, np.abs(pic - pic_copy).max())
        cv2.imwrite("dog_filed" + str(epoch) + ".jpg", img=np.uint8(pic * 255))
    pic_copy = pic.copy()
    if epoch < epochs - 1:
        for i in range(3):
            pic[:, :, i] = BSCB_inpaint(
                pic_copy[:, :, i],
                mask,
                epsilon,
                inpaint_iters,
                anidiffuse_iters,
                delta_ts,
                sensitivites,
                diffuse_coef,
            )
pic = np.uint8(pic * 255)

# specturm_show(pic)
# spectrum_show(pic, "title")
