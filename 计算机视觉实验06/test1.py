import cv2
import numpy as np

# 读取图片
img = cv2.imread("./imgs/cat.jpg")

# 几何变换
rows, cols, _ = img.shape
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
img_rotate = cv2.warpAffine(img, M, (cols, rows))
img_flip = cv2.flip(img, 1)

# 颜色变换
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_h, img_s, img_v = cv2.split(img_hsv)
img_h = cv2.equalizeHist(img_h)
img_s = cv2.equalizeHist(img_s)
img_v = cv2.equalizeHist(img_v)
img_hsv_eq = cv2.merge([img_h, img_s, img_v])
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray_eq = cv2.equalizeHist(img_gray)

# 显示结果
cv2.imshow("Original", img)
cv2.imshow("Rotated", img_rotate)
cv2.imshow("Flipped", img_flip)
cv2.imshow("HSV Equalized", cv2.cvtColor(img_hsv_eq, cv2.COLOR_HSV2BGR))
cv2.imshow("Gray Equalized", cv2.cvtColor(img_gray_eq, cv2.COLOR_GRAY2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
