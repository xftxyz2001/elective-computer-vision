from pptcode import *
import cv2
import numpy as np


# 读取目标图片
target = cv2.imread("./imgs/scenery4.jpg")
# 读取模板图片
template = cv2.imread("./imgs/scenery4-0.jpg")
# 获得模板图片的高宽尺寸
theight, twidth = template.shape[:2]
# 执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED
result = cv2.matchTemplate(target, template, cv2.TM_SQDIFF_NORMED)
# 归一化处理
cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
# 寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)


# 截取
width, height = twidth, theight  # 显示卡片的宽和高
pts1 = np.float32(
    [
        [min_loc[0], min_loc[1]],
        [min_loc[0] + twidth, min_loc[1]],
        [min_loc[0], min_loc[1] + theight],
        [min_loc[0] + twidth, min_loc[1] + theight],
    ]
)  # 截取对片中的哪个区域
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])  # 定义显示的卡片的坐标
matrix = cv2.getPerspectiveTransform(pts1, pts2)  # 两个区域坐标绑定
imgOutput = cv2.warpPerspective(target, matrix, (width, height))  # 转换为图片


# 匹配值转换为字符串
# 对于cv2.TM_SQDIFF及cv2.TM_SQDIFF_NORMED方法min_val越趋近与0匹配度越好，匹配位置取min_loc
# 对于其他方法max_val越趋近于1匹配度越好，匹配位置取max_loc
strmin_val = str(min_val)
# 绘制矩形边框，将匹配区域标注出来
# min_loc：矩形定点
# (min_loc[0]+twidth,min_loc[1]+theight)：矩形的宽高
# (0,0,225)：矩形的边框颜色；2：矩形边框宽度
cv2.rectangle(
    target, min_loc, (min_loc[0] + twidth, min_loc[1] + theight), (0, 0, 225), 2
)


sim1 = LpS(template, imgOutput)
print("LpS:", sim1)
sim2 = diffS(template, imgOutput)
print("diffS:", sim2)
sim3 = cosin(template, imgOutput)
print("cosin:", sim3)


# 显示结果,并将匹配值显示在标题栏上
cv2.imshow("MatchResult----MatchingValue=" + strmin_val, target)
cv2.waitKey()
cv2.destroyAllWindows()
