from pptcode import *

img = cv2.imread("img/1.jpg")
example(img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 最大类间差法（大津算法OTSU）
print("------最大类间差法（大津算法OTSU）------")
OTSU(img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 最小误差算法（双峰法）
print("------最小误差算法（双峰法）------")
threshTwoPeaks(img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 迭代法
print("------迭代法------")
threshIterative(img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 最大熵算法
print("------最大熵算法------")
threshEntroy(img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 自适应动态阈值分割
print("------自适应动态阈值分割------")
adaptiveThresh(img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 基于距离变换的分水岭算法
print("------基于距离变换的分水岭算法------")
watershed(img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 基于梯度的分水岭算法
print("------基于梯度的分水岭算法------")
watershedG(img)
cv2.waitKey(0)
cv2.destroyAllWindows()
