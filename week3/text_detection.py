import os
from glob import glob
import cv2
from textbox_removal import TextBoxRemoval
from paintings_count import computeEdgesToCountPaintings, countNumberPaintingsBasedOnEdges, splitImageInTwo
import heapq

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract


def jaccard_distance(str1, str2):
    str1 = set(str1.split())
    str2 = set(str2.split())
    if len(str1 | str2) == 0: return 0.0
    else: return float(len(str1 & str2)) / len(str1 | str2)


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
    cropped_text = cv2.medianBlur(cropped_text, 3)
    cropped_text = cv2.GaussianBlur(cropped_text, (5, 5), 0)
    text_img_path = img_path.replace(".jpg", "_text.png")
    cv2.imwrite(text_img_path, cropped_text)
    text = detect_text(text_img_path)
    print("TEXT FOUND: ", text)
    return text


def get_texts_from_image(img_path, need_to_cut=True):
    print("Work on img", img_path)
    img_paths = []
    if need_to_cut:
        mask = computeEdgesToCountPaintings(img_path)
        count, cut = countNumberPaintingsBasedOnEdges(mask, img_path)
        if count == 2:
            imgs = splitImageInTwo(img_path, cut)
            for i, img in enumerate(imgs, 1):
                cut_img_path = img_path.replace(".jpg", "_cut_text_" + str(i) + ".jpg")
                cv2.imwrite(cut_img_path, img)
                img_paths.append(cut_img_path)
        else:
            img_paths = [img_path]
    else:
        img_paths = [img_path]

    texts = []
    for path in img_paths:
        text = get_text_single_painting_img(path)
        texts.append(text)
    return texts


def write_all_text_files(img_folder, need_to_cut=True):
    for img_path in sorted(glob(os.path.join(img_folder, "*05.jpg")))[:]:
        texts = get_texts_from_image(img_path, need_to_cut)
        text_path = img_path.replace(".jpg", ".txt")
        with open(text_path, 'w') as w:
            for text in texts:
                w.write(text)
            w.close()
    return 0


def text_based_search(img_folder, need_to_cut=True):
    def compare(bbdd_path, text):
        with open(bbdd_path, 'r', encoding='utf-8') as r:
            bbdd_text = r.read()
            author_start = bbdd_text.find('(') + 2
            author_end = bbdd_text.find(',') - 1
            author = bbdd_text[author_start: author_end]
        print(author + " vs " + text)
        return jaccard_distance(author, text)

    database_folder = "../database"
    k = 2
    results = []
    for img_path in sorted(glob(os.path.join(img_folder, "*.jpg")))[:]:
        result = []
        texts = get_texts_from_image(img_path, need_to_cut)
        for text in texts:
            print("Text: " + text)
            f = lambda bbdd: compare(bbdd, text)
            bbdd_paths = sorted(glob(os.path.join(database_folder, "*.txt")))
            most_similar = heapq.nlargest(k, bbdd_paths, key=f)

            result.append(most_similar)
        print(result)
        results.append(result)

    return results


if __name__ == '__main__':
    imgs_folder = "../qsd1_w3"
    need_to_cut = False
    write_all_text_files(imgs_folder, need_to_cut)
    #text_based_search(imgs_folder, need_to_cut)
