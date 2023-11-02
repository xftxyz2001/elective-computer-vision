import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def read_image(file_path, library):
    if library.lower() == "matplotlib":
        img = plt.imread(file_path)
    elif library.lower() == "pil":
        img = Image.open(file_path)
    elif library.lower() == "cv2":
        img = cv2.imread(file_path)
    else:
        raise ValueError("Invalid library name")
    return img


def matplotlib2PIL(img):
    return Image.fromarray((img * 255).astype("uint8"))


def PIL2matplotlib(img):
    return np.asarray(img) / 255


def matplotlib2cv2(img):
    return (img * 255).astype("uint8")


def cv22matplotlib(img):
    return img / 255


def PIL2cv2(img):
    return np.asarray(img)


def cv22PIL(img):
    return Image.fromarray(img)


def convert_image(img, from_library, to_library):
    if from_library.lower() == "matplotlib" and to_library.lower() == "pil":
        return matplotlib2PIL(img)
    elif from_library.lower() == "pil" and to_library.lower() == "matplotlib":
        return PIL2matplotlib(img)
    elif from_library.lower() == "matplotlib" and to_library.lower() == "cv2":
        return matplotlib2cv2(img)
    elif from_library.lower() == "cv2" and to_library.lower() == "matplotlib":
        return cv22matplotlib(img)
    elif from_library.lower() == "pil" and to_library.lower() == "cv2":
        return PIL2cv2(img)
    elif from_library.lower() == "cv2" and to_library.lower() == "pil":
        return cv22PIL(img)
    else:
        raise ValueError("Invalid library name")


def save_image(img, file_path, library):
    if library.lower() == "matplotlib":
        plt.imsave(file_path, img)
    elif library.lower() == "pil":
        img.save(file_path)
    elif library.lower() == "cv2":
        cv2.imwrite(file_path, img)
    else:
        raise ValueError("Invalid library name")


def show_image(img, library):
    if library.lower() == "matplotlib":
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    elif library.lower() == "pil":
        img.show()
    elif library.lower() == "cv2":
        cv2.imshow("img", img)
        while cv2.waitKey(10) != 27:  # loop if not get ESC
            if cv2.getWindowProperty("img", cv2.WND_PROP_VISIBLE) <= 0:
                break
        cv2.destroyAllWindows()
    else:
        raise ValueError("Invalid library name")


# test
if __name__ == "__main__":
    # 测试图像读取
    img_cv2 = read_image("eye.jpg", "cv2")
    print(type(img_cv2))  # <class 'numpy.ndarray'>

    img_pil = read_image("eye.jpg", "pil")
    print(type(img_pil))  # <class 'PIL.JpegImagePlugin.JpegImageFile'>

    img_matplotlib = read_image("eye.jpg", "matplotlib")
    print(type(img_matplotlib))  # <class 'numpy.ndarray'>
    print()

    # 测试图像转换
    img_cv2_to_pil = convert_image(img_cv2, "cv2", "pil")
    print(type(img_cv2_to_pil))  # <class 'PIL.Image.Image'>

    img_pil_to_cv2 = convert_image(img_pil, "pil", "cv2")
    print(type(img_pil_to_cv2))  # <class 'numpy.ndarray'>

    img_matplotlib_to_pil = convert_image(img_matplotlib, "matplotlib", "pil")
    print(type(img_matplotlib_to_pil))  # <class 'PIL.Image.Image'>

    img_pil_to_matplotlib = convert_image(img_pil, "pil", "matplotlib")
    print(type(img_pil_to_matplotlib))  # <class 'numpy.ndarray'>

    img_cv2_to_matplotlib = convert_image(img_cv2, "cv2", "matplotlib")
    print(type(img_cv2_to_matplotlib))  # <class 'numpy.ndarray'>

    img_matplotlib_to_cv2 = convert_image(img_matplotlib, "matplotlib", "cv2")
    print(type(img_matplotlib_to_cv2))  # <class 'numpy.ndarray'>

    # 测试图像显示
    show_image(img_cv2, "cv2")
    show_image(img_pil, "pil")
    show_image(img_matplotlib, "matplotlib")

    # 测试图像保存
    save_image(img_cv2, "eye_cv2.jpg", "cv2")
    save_image(img_pil, "eye_pil.jpg", "pil")
    save_image(img_matplotlib, "eye_matplotlib.jpg", "matplotlib")
