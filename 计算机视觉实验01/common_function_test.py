import cv2
from PIL import Image


def test_cv2_grayscale():
    # 加载图像
    img = cv2.imread("eye.jpg")

    # 测试灰度化方法
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert gray.shape == (img.shape[0], img.shape[1])


def test_cv2_threshold():
    # 加载图像
    img = cv2.imread("eye.jpg")

    # 测试阈值化方法
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    assert thresh.shape == (img.shape[0], img.shape[1])


def test_cv2_resize():
    # 加载图像
    img = cv2.imread("eye.jpg")

    # 测试缩放方法
    resized = cv2.resize(img, (100, 100))
    assert resized.shape == (100, 100, 3)


def test_cv2_rotate():
    # 加载图像
    img = cv2.imread("eye.jpg")

    # 测试旋转方法
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))
    assert rotated.shape == (rows, cols, 3)


def test_cv2_crop():
    # 加载图像
    img = cv2.imread("eye.jpg")

    # 测试裁剪方法
    rows, cols, _ = img.shape
    cropped = img[rows // 4 : rows // 2, cols // 4 : cols // 2]
    assert cropped.shape == (rows // 4, cols // 4, 3)


def test_pil_grayscale():
    # 加载图像
    img = Image.open("eye.jpg")

    # 测试灰度化方法
    gray = img.convert("L")
    assert gray.size == img.size


def test_pil_threshold():
    # 加载图像
    img = Image.open("eye.jpg")

    # 测试阈值化方法
    gray = img.convert("L")
    thresh = gray.point(lambda x: 255 if x > 127 else 0)
    assert thresh.size == img.size


def test_pil_resize():
    # 加载图像
    img = Image.open("eye.jpg")

    # 测试缩放方法
    resized = img.resize((100, 100))
    assert resized.size == (100, 100)


def test_pil_rotate():
    # 加载图像
    img = Image.open("eye.jpg")

    # 测试旋转方法
    rotated = img.rotate(45)
    assert rotated.size == img.size


def test_pil_crop():
    # 加载图像
    img = Image.open("eye.jpg")

    # 测试裁剪方法
    cropped = img.crop(
        (img.width // 4, img.height // 4, img.width // 2, img.height // 2)
    )
    assert cropped.size == (img.width // 4, img.height // 4)


if __name__ == "__main__":
    test_cv2_grayscale()
    test_cv2_threshold()
    test_cv2_resize()
    test_cv2_rotate()
    test_cv2_crop()
    test_pil_grayscale()
    test_pil_threshold()
    test_pil_resize()
    test_pil_rotate()
    test_pil_crop()
    print("所有测试通过！")
