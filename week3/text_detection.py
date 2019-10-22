import os
from glob import glob
import cv2
from textbox_removal import TextBoxRemoval
from paintings_count import computeEdgesToCountPaintings, countNumberPaintingsBasedOnEdges, splitImageInTwo

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

def jaccard_distance(str1, str2):
    str1 = set(str1.split())
    str2 = set(str2.split())
    return float(len(str1 & str2)) / len(str1 | str2)


def compare_text(query_text, bbdd_text):
    return jaccard_distance(query_text, bbdd_text)


def detect_text(text_img_path):
    return pytesseract.image_to_string(Image.open(text_img_path))


def get_text_single_painting_img(img_path):
    print("get text from single img " + img_path)
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
        print("Work on img", img_path)
        mask = computeEdgesToCountPaintings(img_path)
        count, cut = countNumberPaintingsBasedOnEdges(mask, img_path)
        img_paths = []
        if count == 2:
            imgs = splitImageInTwo(img_path, cut)
            for i, img in enumerate(imgs, 1):
                cut_img_path = img_path.replace(".jpg", "_cut_text_" + str(i) + ".jpg")
                cv2.imwrite(cut_img_path, img)
                img_paths.append(cut_img_path)
        else:
            img_paths = [img_path]

        text_path = img_path.replace(".jpg", ".txt")
        with open(text_path, 'w') as w:
            for path in img_paths:
                text = get_text_single_painting_img(path)
                print(text)
                w.write(text)
            w.close()
    return 0


#def text_based_search(img_folder):


if __name__ == '__main__':
    imgs_folder = "../qsd2_w3"
    main(imgs_folder)
