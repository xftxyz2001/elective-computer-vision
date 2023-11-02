from util import *
import cv2

# 【1】选两张合适的图片，自定义mask，通过简单代数运算融合图片，融合可在HSL等颜色空间进行。需选择合适的代数运算，使得融合效果较好。
catImg = cv2.imread("./imgs/cat.jpg")
dogImg = cv2.imread("./imgs/dog.jpg")

# alphaBlend
# subtractBlend
# multiplyBlend
# multiplyBlend0
# divideBlend
# maxBlend
# minBlend
# multiplyBlend1
# multiplyBlend2
cvBGRBlend0(catImg, dogImg, "cvBGRBlend0-alphaBlend", f=alphaBlend)
cvBGRBlend0(catImg, dogImg, "cvBGRBlend0-subtractBlend", f=subtractBlend)
cvBGRBlend0(catImg, dogImg, "cvBGRBlend0-multiplyBlend", f=multiplyBlend)
cvBGRBlend0(catImg, dogImg, "cvBGRBlend0-multiplyBlend0", f=multiplyBlend0)
cvBGRBlend0(catImg, dogImg, "cvBGRBlend0-divideBlend", f=divideBlend)
cvBGRBlend0(catImg, dogImg, "cvBGRBlend0-maxBlend", f=maxBlend)
cvBGRBlend0(catImg, dogImg, "cvBGRBlend0-minBlend", f=minBlend)
cvBGRBlend0(catImg, dogImg, "cvBGRBlend0-multiplyBlend1", f=multiplyBlend1)
cvBGRBlend0(catImg, dogImg, "cvBGRBlend0-multiplyBlend2", f=multiplyBlend2)

cvBGRBlend0(dogImg, catImg, "cvBGRBlend0-alphaBlend-r", f=alphaBlend)
cvBGRBlend0(dogImg, catImg, "cvBGRBlend0-subtractBlend-r", f=subtractBlend)
cvBGRBlend0(dogImg, catImg, "cvBGRBlend0-multiplyBlend-r", f=multiplyBlend)
cvBGRBlend0(dogImg, catImg, "cvBGRBlend0-multiplyBlend0-r", f=multiplyBlend0)
cvBGRBlend0(dogImg, catImg, "cvBGRBlend0-divideBlend-r", f=divideBlend)
cvBGRBlend0(dogImg, catImg, "cvBGRBlend0-maxBlend-r", f=maxBlend)
cvBGRBlend0(dogImg, catImg, "cvBGRBlend0-minBlend-r", f=minBlend)
cvBGRBlend0(dogImg, catImg, "cvBGRBlend0-multiplyBlend1-r", f=multiplyBlend1)
cvBGRBlend0(dogImg, catImg, "cvBGRBlend0-multiplyBlend2-r", f=multiplyBlend2)

cvHLSBlend0(catImg, dogImg, "cvHLSBlend0-alphaBlend", f=alphaBlend)
cvHLSBlend0(catImg, dogImg, "cvHLSBlend0-subtractBlend", f=subtractBlend)
cvHLSBlend0(catImg, dogImg, "cvHLSBlend0-multiplyBlend", f=multiplyBlend)
cvHLSBlend0(catImg, dogImg, "cvHLSBlend0-multiplyBlend0", f=multiplyBlend0)
cvHLSBlend0(catImg, dogImg, "cvHLSBlend0-divideBlend", f=divideBlend)
cvHLSBlend0(catImg, dogImg, "cvHLSBlend0-maxBlend", f=maxBlend)
cvHLSBlend0(catImg, dogImg, "cvHLSBlend0-minBlend", f=minBlend)
cvHLSBlend0(catImg, dogImg, "cvHLSBlend0-multiplyBlend1", f=multiplyBlend1)
cvHLSBlend0(catImg, dogImg, "cvHLSBlend0-multiplyBlend2", f=multiplyBlend2)

cvLABBlend0(catImg, dogImg, "cvLABBlend0-alphaBlend", f=alphaBlend)
cvLABBlend0(catImg, dogImg, "cvLABBlend0-subtractBlend", f=subtractBlend)
cvLABBlend0(catImg, dogImg, "cvLABBlend0-multiplyBlend", f=multiplyBlend)
cvLABBlend0(catImg, dogImg, "cvLABBlend0-multiplyBlend0", f=multiplyBlend0)
cvLABBlend0(catImg, dogImg, "cvLABBlend0-divideBlend", f=divideBlend)
cvLABBlend0(catImg, dogImg, "cvLABBlend0-maxBlend", f=maxBlend)
cvLABBlend0(catImg, dogImg, "cvLABBlend0-minBlend", f=minBlend)
cvLABBlend0(catImg, dogImg, "cvLABBlend0-multiplyBlend1", f=multiplyBlend1)
cvLABBlend0(catImg, dogImg, "cvLABBlend0-multiplyBlend2", f=multiplyBlend2)

