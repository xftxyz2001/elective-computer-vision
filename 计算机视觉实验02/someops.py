import cv2 as cv
import numpy as np
import cv2.ximgproc

# 读取图像
im = cv.imread("eye.jpg")

# 高斯滤波
gaussian = cv.GaussianBlur(im, (5, 5), 0)

# 中值滤波
median = cv.medianBlur(im, 5)

# 双边滤波
bilateral = cv.bilateralFilter(im, 9, 75, 75)

# 形态学操作
kernel = np.ones((5, 5), np.uint8)
erosion = cv.erode(im, kernel, iterations=1)
dilation = cv.dilate(im, kernel, iterations=1)
opening = cv.morphologyEx(im, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(im, cv.MORPH_CLOSE, kernel)


# 雾去除
def dehaze(image, w=0.95, t0=0.55, Amax=220):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    dc = cv.ximgproc.createFastGlobalSmootherFilter(gray, 15, 200, 200)
    t = dc.filter(gray, gray)
    t = cv.max(t, t0 * 255)
    t = cv.min(t, 255)
    t = t.astype(np.uint8)
    A = dc.filter(image, t)
    A = cv.medianBlur(A, 5)
    A = cv.cvtColor(A, cv.COLOR_BGR2GRAY)  # Convert to single-channel image
    A = np.minimum(A, Amax)
    J = np.zeros_like(image)
    for i in range(3):
        J[:, :, i] = (image[:, :, i] - A) / cv.max(w, t / 255.0) + A
    J = cv.max(J, 0)
    J = cv.min(J, 255)
    J = J.astype(np.uint8)
    return J


dehazed = dehaze(im)

# 显示结果
cv.imshow("origin", im)
cv.imshow("gaussian", gaussian)
cv.imshow("median", median)
cv.imshow("bilateral", bilateral)
cv.imshow("erosion", erosion)
cv.imshow("dilation", dilation)
cv.imshow("opening", opening)
cv.imshow("closing", closing)
cv.imshow("dehazed", dehazed)
cv.waitKey(0)
cv.destroyAllWindows()
