import cv2
import pptcode

# 加载测试图片
# img = cv2.imread("test.png")
img = cv2.imread("2.jpg")
cv2.imshow("Original", img)

# 测试边缘检测函数
roberts = pptcode.getRoberts(img)  # Roberts算子
prewitt = pptcode.getPrewitt(img)  # Prewitt算子
sobel = pptcode.getSobel(img)  # Sobel算子
laplacian = pptcode.getLaplacian(img)  # Laplacian算子
log = pptcode.getLoG(img)  # LoG算子
canny = cv2.Canny(img, 100, 200)  # Canny边缘检测

# 测试角点检测函数
orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(img, None)

# 测试Hough变换函数
line_detection = pptcode.line_detection(img)  # 标准Hough变换
line_detectionP = pptcode.line_detectionP(img)  # 累计概率Hough变换
HoughCircles = pptcode.HoughCircles(img)  # Hough圆变换

# # 显示结果
# cv2.imshow("Roberts", roberts)
# cv2.imshow("Prewitt", prewitt)
# cv2.imshow("Sobel", sobel)
# cv2.imshow("Laplacian", laplacian)
# cv2.imshow("LoG", log)
# cv2.imshow("Canny", canny)
# cv2.imshow("ORB", cv2.drawKeypoints(img, kp, None))
# cv2.imshow("Line Detection", line_detection)
# cv2.imshow("Line Detection P", line_detectionP)
# cv2.imshow("Hough Circles", HoughCircles)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# 显示结果
cv2.imshow("Roberts", roberts)
cv2.waitKey(0)
cv2.imshow("Prewitt", prewitt)
cv2.waitKey(0)
cv2.imshow("Sobel", sobel)
cv2.waitKey(0)
cv2.imshow("Laplacian", laplacian)
cv2.waitKey(0)
cv2.imshow("LoG", log)
cv2.waitKey(0)
cv2.imshow("Canny", canny)
cv2.waitKey(0)
cv2.imshow("ORB", cv2.drawKeypoints(img, kp, None))
cv2.waitKey(0)
cv2.imshow("Line Detection", line_detection)
cv2.waitKey(0)
cv2.imshow("Line Detection P", line_detectionP)
cv2.waitKey(0)
cv2.imshow("Hough Circles", HoughCircles)
cv2.waitKey(0)
cv2.destroyAllWindows()
