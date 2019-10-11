import os
from enum import Enum
from glob import glob
import cv2
import numpy as np
from background_removal import BackgroundMask2

CLOSING_KERNEL_SIZE = 10
OPENING_KERNEL_SIZE = 40
THRESHOLD_TO_CONSIDER_SWITCH_TO_WHITE_REGION = 0.6
THRESHOLD_TO_CONSIDER_SWITCH_TO_BLACK_REGION = 0.5
RESIZE_SHAPE = (400, 400)


def computeMaskToCountPaintings(img_path):
    closing_kernel_size = CLOSING_KERNEL_SIZE
    closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
    opening_kernel_size = OPENING_KERNEL_SIZE
    opening_kernel = np.ones((opening_kernel_size, opening_kernel_size), np.uint8)

    print("\nComputing mask with method 2 for img_path: ", img_path)
    img = cv2.imread(img_path)
    orig_shape = img.shape
    img = cv2.resize(img, RESIZE_SHAPE, interpolation=cv2.INTER_AREA)
    mask = BackgroundMask2(img)

    print("\nApplying morphology operations in mask corresponding to: ", img_path)
    filtered_mask = mask
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, closing_kernel)
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, opening_kernel)
    mask = cv2.resize(mask, (orig_shape[1], orig_shape[0]))
    cv2.imwrite(img_path.replace(".jpg", "_old_mask.png"), mask)
    cv2.imwrite(img_path.replace(".jpg", "_count_mask.png"), filtered_mask)
    filtered_mask = cv2.resize(filtered_mask, RESIZE_SHAPE)  # (x_length, y_length)

    return filtered_mask


def countNumberPaintingsBasedOnMask(mask, img_path):
    print("\nCounting number of paintings in img_path: ", img_path)
    height = mask.shape[0]
    width = mask.shape[1]
    white_threshold = height * THRESHOLD_TO_CONSIDER_SWITCH_TO_WHITE_REGION
    black_threshold = height * THRESHOLD_TO_CONSIDER_SWITCH_TO_BLACK_REGION

    number_of_white_predominant_regions = 0
    pixel_count_end_of_first_painting = -1
    pixel_count_start_of_second_painting = -1
    current_region = colorRegion.BLACK
    print("Looping on width: ", width)
    for i in range(0, width):
        vertical_pixel_line = mask[:, i]
        assert len(vertical_pixel_line) == height
        white_pixel_count = np.count_nonzero(vertical_pixel_line)

        if current_region == colorRegion.BLACK and white_pixel_count >= white_threshold:
            current_region = colorRegion.WHITE
            number_of_white_predominant_regions += 1
            if pixel_count_end_of_first_painting > 0:
                pixel_count_start_of_second_painting = i
        elif current_region == colorRegion.WHITE and white_pixel_count < black_threshold:
            if pixel_count_start_of_second_painting <= 0:
                pixel_count_end_of_first_painting = i
            current_region = colorRegion.BLACK

    print("Count of paintings: " + str(number_of_white_predominant_regions))
    pixel_count_cut = pixel_count_end_of_first_painting + ((pixel_count_start_of_second_painting - pixel_count_end_of_first_painting) / 2)
    return number_of_white_predominant_regions, pixel_count_cut


def splitImageInTwo(img_path, cutPoint):
    print("\nSplitting img_path: " + img_path + " at cutPoint " + str(cutPoint))
    img = cv2.imread(img_path)
    orig_shape = img.shape
    orig_width = orig_shape[1]
    orig_height = orig_shape[0]
    scaled_cut_point = int((cutPoint / RESIZE_SHAPE[1]) * orig_width)
    img1 = img[:, 0: scaled_cut_point]
    img2 = img[:, scaled_cut_point: orig_width]
    images = [img1, img2]
    cv2.imwrite(img_path.replace(".jpg", "_cut1.png"), img1)
    cv2.imwrite(img_path.replace(".jpg", "_cut2.png"), img2)
    return images


class colorRegion(Enum):
    BLACK = 0,
    WHITE = 1


def getListOfPaintings(img_folder):
    output = []
    for img_path in sorted(glob(os.path.join(img_folder, "*.jpg")))[:]:
        mask = computeMaskToCountPaingings(img_path)
        count, cut = countNumberPaintingsBasedOnMask(mask, img_path)
        if count == 2:
            print("-- Cut at pixel: ", str(cut))
            output.append(splitImageInTwo(img_path, cut))
        else:
            output.append(cv2.imread(img_path))


def main(img_folder):
    error_images = []
    for img_path in sorted(glob(os.path.join(img_folder, "*.jpg")))[:]:
        expected = 1
        if img_path in ["../qsd2_w2/00001.jpg", "../qsd2_w2/00004.jpg", "../qsd2_w2/00009.jpg", "../qsd2_w2/00017.jpg",
                        "../qsd2_w2/00018.jpg", "../qsd2_w2/00025.jpg", "../qsd2_w2/00028.jpg"]:
            expected = 2

        mask = computeMaskToCountPaingings(img_path)
        count, cut = countNumberPaintingsBasedOnMask(mask, img_path)

        if count == expected:
            print("-- Good guess for image: ", img_path)
        else:
            error_images.append(img_path)
            print("-- Error for image " + img_path + ", expected: " + str(expected) + ", actual: " + str(count))

        if count == 2:
            print("-- Cut at pixel: ", str(cut))
            splitImageInTwo(img_path, cut)

    print("-- Pictures with wrong guesses --")
    print(error_images)


if __name__ == '__main__':
    main("../qsd2_w2")

'''Anotate results:
Note that we prefer to predict 2 in one, that missing a picture
For params: 0.6, 0.5 => 00000.jpg bad guess counts 2 instead of 1, 00017.jpg bad guess counts 1 instead of 2. Seems hard to correct both
                        since they have two opposite strategies, In case of 0000.jpg if I push the white threshold to 0.7 can be solved,
                        but the problem of 00017.jpg is that the first of the paintings is wrong
For params: 0.7, 0.5 => Fix 00000.jpg but breaks 00004.jpg, 000018 and 00025 and still has 00017 wrong
For params: 0.65, 0.5 => Fix 000018.jpg and 00004.jpg but breaks back 00000.jpg, and still has 00017 and 00018 wrong

With params 0.6, 0.5 => Change the kernel size from (10,10) (40,40) to a (30,30), (40,40) => Bad guess in 4.jpg, 9.jpg, 28.jpg
'''
