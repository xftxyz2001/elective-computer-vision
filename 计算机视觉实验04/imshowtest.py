import cv2

img = cv2.imread("img/1.jpg")
cv2.imshow("窗口标题".encode("gbk").decode(errors="ignore"), img)
cv2.waitKey(0)
