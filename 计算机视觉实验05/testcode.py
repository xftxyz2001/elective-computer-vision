from pptcode import *

# 下面是一些写好的工具函数，可以直接使用
# Lp相似度 LpS(X, Y, p=2.0)
# 差相似度 diffS(X, Y)
# 余弦相似度 cosin(X, Y)
# 古本（Tanimoto）相似度 Tanimoto(X, Y)
# 皮尔逊（Pearson）相似度 correlationN(X, Y)
# 结构相似度（SSIM） ssim(X, Y, data_range=255.0, K=(0.01, 0.03))
# 平均结构相似度（MSSIM） Mssim(img1, img2)
# 峰值信噪比（PSNR） psnr(X, Y)
# 获取直方图、显示结果 GrayHist(img) showHistResult(hist1, hist2)
# Lp相似度 for 直方图 LpS_hist(hist1, hist2, p=2.0)

imagePaths = [
    "imgs/cat.jpg",
    "imgs/catdog.jpg",
    "imgs/coin2.jpg",
    "imgs/coin3.jpg",
    "imgs/dog.jpg",
    "imgs/eye.jpg",
    "imgs/foggy.jpg",
    "imgs/foggybuilding.jpg",
    "imgs/hazelnut.jpg",
    "imgs/magnifier.jpg",
    "imgs/rgb.jpg",
    "imgs/road.jpg",
    "imgs/scenery1.jpg",
    "imgs/scenery2.jpg",
    "imgs/scenery3.jpg",
    "imgs/scenery4-0.jpg",
    "imgs/scenery4.jpg",
    "imgs/scenery5.jpg",
    "imgs/scenery6.jpg",
    "imgs/test1.jpg",
    "imgs/test2.jpg",
]

# 【1】选2-3组图片，使用基于模版的像素匹配定位图片，用3-5种相似性度量，需要输出定位时的相似度。

# 【2】选1-2组图片，使用基于模版的直方图匹配定位图片，用3-5种相似性度量，需要输出定位时的相似度。

# 【3】选1-2组图片，用SIFT或者其他特征匹配方法定位图片，可尝试输出基于特征点的匹配程度(距离或相似度)。
