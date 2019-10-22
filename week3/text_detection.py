import os
from glob import glob
import cv2
from textbox_removal import TextBoxRemoval

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract


def detect_text(text_img_path):
    return pytesseract.image_to_string(Image.open(text_img_path))


def get_text_single_painting_img(img_path):
    img = cv2.imread(img_path)
    mask, textbox = TextBoxRemoval(img)
    bbox = [textbox[0][1], textbox[0][0], textbox[1][1], textbox[1][0]]
    cv2.imwrite(img_path.replace(".jpg", "_textboxmask.png"), mask)
    text_img = cv2.bitwise_and(img, img, mask=~mask)
    cropped_text = text_img[bbox[1]: bbox[3], bbox[0]: bbox[2]]
    text_img_path = img_path.replace(".jpg", "_text.png")
    cv2.imwrite(text_img_path, cropped_text)
    text = detect_text(text_img_path)
    return text


def main(img_folder):
    for img_path in sorted(glob(os.path.join(img_folder, "*.jpg")))[:]:
        print(get_text_single_painting_img(img_path))
    return 0


if __name__ == '__main__':
    imgs_folder = "../qsd1_w3"
    main(imgs_folder)
