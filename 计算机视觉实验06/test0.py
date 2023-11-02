from util import *
import cv2

# 【1】分别选合适的图片，通过灰度变换、直方图规定化、频域滤波、伪彩色着色等方法进行图像增强。

# 【2】联合几何变换类和颜色变换类方法，为同一张图片进行数据扩充，要求扩充到30张以上。

# 【3】选合适的图片，从灰度变换、直方图规定化、空域滤波、频域滤波、同态滤波、伪彩色着色、颜色变换等
# 方法中选择合适的组合联合处理图片，使图片视觉质量得到明显的改善或者呈现出独特的视觉效果。

# 加载图片
dogImg = cv2.imread("./imgs/dog.jpg")
catImg = cv2.imread("./imgs/cat.jpg")

# 展示直方图
showImgHist(dogImg)
showImgHist(catImg)

# 灰度变换
# TwoSegment0
# FourSegment0
cvBGRAdjust0(dogImg, "dogImg0-TwoSegment0", f=TwoSegment0)
cvBGRAdjust0(dogImg, "dogImg0-FourSegment0", f=FourSegment0)

# 非线性灰度变换
# pow0
# pow1
# sigmoid0
# sigmoid1
# logic0
# logic1
# s0
cvBGRAdjust1(dogImg, "dogImg1-pow0", f=pow0)
cvBGRAdjust1(dogImg, "dogImg1-pow1", f=pow1)
cvBGRAdjust1(dogImg, "dogImg1-sigmoid0", f=sigmoid0)
cvBGRAdjust1(dogImg, "dogImg1-sigmoid1", f=sigmoid1)
cvBGRAdjust1(dogImg, "dogImg1-logic0", f=logic0)
cvBGRAdjust1(dogImg, "dogImg1-logic1", f=logic1)
cvBGRAdjust1(dogImg, "dogImg1-s0", f=s0)

# 直方图规定化
# 参考infer_map给img_org配色
get_hist_match(dogImg, catImg)
get_hist_match(catImg, dogImg)

# 显示频谱图
spectrum_show(dogImg)
spectrum_show(catImg)

# 频域滤波
# cal_distance//
# IdealLowPass
# ButterworthLowPass
# GaussianLowPass
# IdealhighPass
# ButterworthhighPass
# GaussianhighPass
# GaussianhighPassEmphasize
# spectralFilter(dogImg, f=cal_distance)
spectralFilter(dogImg, f=IdealLowPass)
spectralFilter(dogImg, f=ButterworthLowPass)
spectralFilter(dogImg, f=GaussianLowPass)
spectralFilter(dogImg, f=IdealhighPass)
spectralFilter(dogImg, f=ButterworthhighPass)
spectralFilter(dogImg, f=GaussianhighPass)
spectralFilter(dogImg, f=GaussianhighPassEmphasize)

# 同态滤波
homomorphic_filter(dogImg)
homomorphic_filter(catImg)

homomorphic_filter_HSL(dogImg)
homomorphic_filter_HSL(catImg)

# 伪彩色着色
flower = cv2.imread("./imgs/flower.jpg")
pseudocolorFlower(flower)

# 色彩变换
cvHSLAdjust0(dogImg, "dogImg0-HSLAdjust0")
cvHSLAdjust1(dogImg, "dogImg1-HSLAdjust1")
cvHSLAdjust2(dogImg, "dogImg2-HSLAdjust2")
