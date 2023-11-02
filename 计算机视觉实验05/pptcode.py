import cv2
import numpy as np
import matplotlib.pyplot as plt


# Lp相似度
def LpS(X, Y, p=2.0):
    x = np.float64(X.reshape(-1))
    y = np.float64(Y.reshape(-1))
    n = len(x)
    return 1.0 - np.linalg.norm(x - y, p) / 255.0 / n ** (1.0 / p)


# 差相似度
def diffS(X, Y):
    x = np.float64(X.reshape(-1))
    y = np.float64(Y.reshape(-1))
    n = len(x)
    return 1.0 - np.abs(np.sum(x - y)) / 255.0 / n


# 余弦相似度
def cosin(X, Y):
    vector_a = np.float64(X.reshape(-1))
    vector_b = np.float64(Y.reshape(-1))
    num = np.dot(vector_a, vector_b)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    sim = num / denom
    return sim


# 古本（Tanimoto）相似度
def Tanimoto(X, Y):
    vector_a = np.float64(X.reshape(-1))
    vector_b = np.float64(Y.reshape(-1))
    num = np.dot(vector_a, vector_b)
    denom = np.dot(vector_a, vector_a) + np.dot(vector_b, vector_b) - num
    sim = num / denom
    return sim


# 皮尔逊（Pearson）相似度
def correlationN(X, Y):
    x = np.float64(X.reshape(-1))
    y = np.float64(Y.reshape(-1))
    return np.corrcoef(x, y)[0, 1]


# 结构相似度（SSIM）
def ssim(X, Y, data_range=255.0, K=(0.01, 0.03)):
    K1, K2 = K
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = np.mean(X)
    mu2 = np.mean(Y)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu12 = mu1 * mu2
    sigma1_sq = np.mean((X - mu1_sq) ** 2)
    sigma2_sq = np.mean((Y - mu2_sq) ** 2)
    sigma12 = np.mean((X - mu1_sq) * (Y - mu2_sq))
    cs_ = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_ = (2 * mu12 + C1) / (mu1_sq + mu2_sq + C1) * cs_
    return ssim_


# 平均结构相似度（MSSIM）
def Mssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    hws = 5
    sigma = 0.3 * hws
    kernel = cv2.getGaussianKernel(2 * hws + 1, sigma)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[hws:-hws, hws:-hws]
    mu2 = cv2.filter2D(img2, -1, window)[hws:-hws, hws:-hws]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[hws:-hws, hws:-hws] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[hws:-hws, hws:-hws] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[hws:-hws, hws:-hws] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


# 峰值信噪比（PSNR）
def psnr(X, Y):
    x = np.float64(X.reshape(-1))
    y = np.float64(Y.reshape(-1))
    mse = np.mean((x / 255 - y / 255) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * np.log10(1.0**2 / mse)


# 获取直方图、显示结果
def GrayHist(img):
    grayHist = np.zeros(256, dtype=np.uint64)
    for v in range(256):
        grayHist[v] = np.sum(img == v)
    return grayHist


def showHistResult(hist1, hist2):
    plt.plot(hist1, color="b")
    plt.plot(hist2, color="r")
    plt.xlim(0, 256)
    plt.ylim(0, max(np.amax(hist1), np.amax(hist2)))
    plt.xticks([])
    plt.show()


# Lp相似度 for 直方图
def LpS_hist(hist1, hist2, p=2.0):
    x = np.float64(hist1.reshape(-1))
    y = np.float64(hist2.reshape(-1))
    n = len(x)
    x = x / np.sum(x)
    y = y / np.sum(y)
    return 1.0 - np.linalg.norm(x - y, p) / 2 ** (1.0 / p)


# SIFT特征匹配
# https://blog.csdn.net/wu_zhiyuan/article/details/126028766
# https://www.freesion.com/article/7751388844/
