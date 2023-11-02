from util import *
import cv2

dogImg = cv2.imread("./imgs/dog.jpg")
catImg = cv2.imread("./imgs/cat.jpg")

get_hist_match(dogImg, catImg)
get_hist_match(catImg, dogImg)
