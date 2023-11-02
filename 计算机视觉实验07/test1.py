from util import *
import cv2

catImg = cv2.imread("./imgs/cat.jpg")
dogImg = cv2.imread("./imgs/dog.jpg")


# 【2】将实验【1】的图片通过拉普拉斯金字塔分解进行多分辨率融合。
images_list = [
    "cvHLSBlend0-subtractBlend-Blended.jpg",
    "cvBGRBlend0-alphaBlend-Blended.jpg",
    "cvBGRBlend0-multiplyBlend-Blended.jpg",
]
images_list = ["./imgs/cat.jpg", "./imgs/dog.jpg", "./imgs/mask.jpg"]
sequence = np.stack([cv2.imread(name) for name in images_list])
# #拉普拉斯融合
for i in range(1, 10, 3):
    fused_results = laplacian_fusion(sequence, layers_num=i)

    # print(type(fused_results))
    for k, v in fused_results.items():
        cv2.imwrite(k + str(i) + ".jpg", v)


# 【3】将实验【1】的图片通过泊松融合方法进行融合。
spectrum_show(dogImg)
spectrum_show(catImg)

mixed_pywtfuse_mask(catImg, dogImg)
