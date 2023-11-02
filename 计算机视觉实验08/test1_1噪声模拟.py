from util import *

# 【1】任选图片，进行噪声模拟与统计排序滤波器的恢复实验。

# 噪声模拟
dogImg = cv2.imread('./imgs/dog.jpg')

# 高斯狗
gaussian_noiseed = add_gaussian_noise(dogImg)
cv2.imwrite('./noiseed/dog_gaussian_noiseed.jpg', gaussian_noiseed)

# 瑞利狗
rayleigh_noiseed = add_rayleigh_noise(dogImg)
cv2.imwrite('./noiseed/dog_rayleigh_noiseed.jpg', rayleigh_noiseed)

# 伽马狗
gamma_noiseed = add_gamma_noise(dogImg)
cv2.imwrite('./noiseed/dog_gamma_noiseed.jpg', gamma_noiseed)

# 椒盐狗
salt_peppered = add_salt_pepper(dogImg)
cv2.imwrite('./noiseed/dog_salt_peppered.jpg', salt_peppered)


catImg = cv2.imread('./imgs/cat.jpg')

# 高斯猫
gaussian_noiseed = add_gaussian_noise(catImg)
cv2.imwrite('./noiseed/cat_gaussian_noiseed.jpg', gaussian_noiseed)

# 瑞利猫
rayleigh_noiseed = add_rayleigh_noise(catImg)
cv2.imwrite('./noiseed/cat_rayleigh_noiseed.jpg', rayleigh_noiseed)

# 伽马猫
gamma_noiseed = add_gamma_noise(catImg)
cv2.imwrite('./noiseed/cat_gamma_noiseed.jpg', gamma_noiseed)

# 椒盐猫
salt_peppered = add_salt_pepper(catImg)
cv2.imwrite('./noiseed/cat_salt_peppered.jpg', salt_peppered)
