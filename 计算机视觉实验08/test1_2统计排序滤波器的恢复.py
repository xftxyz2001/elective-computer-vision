from util import *


# 【1】任选图片，进行噪声模拟与统计排序滤波器的恢复实验。

# 统计排序滤波器的恢复
dog_gaussian_noiseed = cv2.imread("./noiseed/dog_gaussian_noiseed.jpg")
dog_rayleigh_noiseed = cv2.imread("./noiseed/dog_rayleigh_noiseed.jpg")
dog_gamma_noiseed = cv2.imread("./noiseed/dog_gamma_noiseed.jpg")
dog_salt_peppered = cv2.imread("./noiseed/dog_salt_peppered.jpg")

# 几何均值滤波
fixed_dog_gaussian_noiseed = rgbGemoetricMean(dog_gaussian_noiseed)
cv2.imwrite("./fixed/GemoetricMean-dog_gaussian_noiseed.jpg",
            fixed_dog_gaussian_noiseed)

fixed_dog_rayleigh_noiseed = rgbGemoetricMean(dog_rayleigh_noiseed)
cv2.imwrite("./fixed/GemoetricMean-dog_rayleigh_noiseed.jpg",
            fixed_dog_rayleigh_noiseed)

fixed_dog_gamma_noiseed = rgbGemoetricMean(dog_gamma_noiseed)
cv2.imwrite("./fixed/GemoetricMean-dog_gamma_noiseed.jpg",
            fixed_dog_gamma_noiseed)

fixed_dog_salt_peppered = rgbGemoetricMean(dog_salt_peppered)
cv2.imwrite("./fixed/GemoetricMean-dog_salt_peppered.jpg",
            fixed_dog_salt_peppered)


# 谐波均值滤波
fixed_dog_gaussian_noiseed = rgbHarmonicMean(dog_gaussian_noiseed)
cv2.imwrite("./fixed/HarmonicMean-dog_gaussian_noiseed.jpg",
            fixed_dog_gaussian_noiseed)


fixed_dog_rayleigh_noiseed = rgbHarmonicMean(dog_rayleigh_noiseed)
cv2.imwrite("./fixed/HarmonicMean-dog_rayleigh_noiseed.jpg",
            fixed_dog_rayleigh_noiseed)

fixed_dog_gamma_noiseed = rgbHarmonicMean(dog_gamma_noiseed)
cv2.imwrite("./fixed/HarmonicMean-dog_gamma_noiseed.jpg",
            fixed_dog_gamma_noiseed)

fixed_dog_salt_peppered = rgbHarmonicMean(dog_salt_peppered)
cv2.imwrite("./fixed/HarmonicMean-dog_salt_peppered.jpg",
            fixed_dog_salt_peppered)


# 逆谐波均值滤波
fixed_dog_gaussian_noiseed = rgbContra_harmonicMean(dog_gaussian_noiseed, 1.5)
cv2.imwrite("./fixed/InverseHarmonicMean-dog_gaussian_noiseed.jpg",
            fixed_dog_gaussian_noiseed)

fixed_dog_rayleigh_noiseed = rgbContra_harmonicMean(dog_rayleigh_noiseed, 1.5)
cv2.imwrite("./fixed/InverseHarmonicMean-dog_rayleigh_noiseed.jpg",
            fixed_dog_rayleigh_noiseed)

fixed_dog_gamma_noiseed = rgbContra_harmonicMean(dog_gamma_noiseed, 1.5)
cv2.imwrite("./fixed/InverseHarmonicMean-dog_gamma_noiseed.jpg",
            fixed_dog_gamma_noiseed)

fixed_dog_salt_peppered = rgbContra_harmonicMean(dog_salt_peppered, 1.5)
cv2.imwrite("./fixed/InverseHarmonicMean-dog_salt_peppered.jpg",
            fixed_dog_salt_peppered)


# *--------------------------------------------------------------*
# cat_gamma_noiseed = cv2.imread("./noiseed/cat_gamma_noiseed.jpg")
# cat_gaussian_noiseed = cv2.imread("./noiseed/cat_gaussian_noiseed.jpg")
# cat_rayleigh_noiseed = cv2.imread("./noiseed/cat_rayleigh_noiseed.jpg")
# cat_salt_peppered = cv2.imread("./noiseed/cat_salt_peppered.jpg")
