from PIL import Image

img = Image.open("eye.jpg")
imgGray = img.convert("L")

img.show()
imgGray.show()

img.save("img_copy.jpg")
imgGray.save("img_gray.jpg")

# --------------------------------------------------
import cv2

img = cv2.imread("eye.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("img", img)
while cv2.waitKey(10) != 27:  # loop if not get ESC
    if cv2.getWindowProperty("img", cv2.WND_PROP_VISIBLE) <= 0:
        break
cv2.destroyAllWindows()

cv2.imshow("img", img_gray)
while cv2.waitKey(10) != 27:  # loop if not get ESC
    if cv2.getWindowProperty("img", cv2.WND_PROP_VISIBLE) <= 0:
        break
cv2.destroyAllWindows()

cv2.imwrite("img_copy.jpg", img)
cv2.imwrite("img_gray.jpg", img_gray)


# --------------------------------------------------
import pylab as plt

img = plt.imread("eye.jpg")
r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
img_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

plt.imshow(img)
plt.axis("off")
plt.savefig("img_copy.jpg", bbox_inches="tight", pad_inches=0)
plt.show()

plt.imshow(img_gray, cmap="gray")
plt.axis("off")
plt.savefig("img_gray.jpg", bbox_inches="tight", pad_inches=0)
plt.show()
