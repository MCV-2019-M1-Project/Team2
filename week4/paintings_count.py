import os
from enum import Enum
from glob import glob
import cv2
import numpy as np
from background_removal import BackgroundMask2

# Parameters MASK method
CLOSING_KERNEL_SIZE = 10
OPENING_KERNEL_SIZE = 40
THRESHOLD_TO_CONSIDER_SWITCH_TO_WHITE_REGION = 0.6
THRESHOLD_TO_CONSIDER_SWITCH_TO_BLACK_REGION = 0.5
RESIZE_SHAPE = (400, 400)

# Parameters EDGES method
MINIMUM_BLACK_WIDTH = 0.1
MINIMUM_WHITE_WIDTH = 0.1


def computeMaskToCountPaintings(img):
    closing_kernel_size = CLOSING_KERNEL_SIZE
    closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
    opening_kernel_size = OPENING_KERNEL_SIZE
    opening_kernel = np.ones((opening_kernel_size, opening_kernel_size), np.uint8)

    print("\nComputing mask with method 2")
    orig_shape = img.shape
    img = cv2.resize(img, RESIZE_SHAPE, interpolation=cv2.INTER_AREA)
    mask = BackgroundMask2(img)

    print("\nApplying morphology operations in mask")
    filtered_mask = mask
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, closing_kernel)
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, opening_kernel)
    mask = cv2.resize(mask, (orig_shape[1], orig_shape[0]))
    filtered_mask = cv2.resize(filtered_mask, RESIZE_SHAPE)  # (x_length, y_length)

    return filtered_mask

def computeEdgesToCountPaintings(img):
    # Read image and resize
    original_size = img.shape[:2]
    img = cv2.resize(img,(1000,1000))

    # Obtain gray level image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.imwrite(r"C:\Users\PC\Documents\Roger\Master\M1\Project\Week3\tests_folder\testpaintcount.png",gray)

    # Obtain edges through Canny algorithm
    edges = cv2.Canny(gray,50,150,apertureSize=3)

    # Dilate edges and resize
    kernel = np.ones((3,3),np.uint8)
    edges = cv2.dilate(edges,kernel,iterations=1)
    edges = cv2.resize(edges,(original_size[1],original_size[0]))

    return edges

def countNumberPaintingsBasedOnMask(mask, img):
    print("\nCounting number of paintings")
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
    
    orig_width = img.shape[1]
    pixel_count_cut = int((pixel_count_cut / RESIZE_SHAPE[1]) * orig_width)

    return number_of_white_predominant_regions, pixel_count_cut

def countNumberPaintingsBasedOnEdges(mask, img):
    print("\nCounting number of paintings")
    height = mask.shape[0]
    width = mask.shape[1]
    white_threshold = 0
    black_threshold = 1

    black_region_length = 0
    white_region_length = 0
    number_of_white_predominant_regions = 0
    pixel_count_end_of_first_painting = -1
    pixel_count_start_of_second_painting = -1
    current_region = colorRegion.BLACK
    print("Looping on width: ", width)
    for i in range(0, width):
        vertical_pixel_line = mask[:, i]
        assert len(vertical_pixel_line) == height
        white_pixel_count = np.count_nonzero(vertical_pixel_line)

        if current_region == colorRegion.BLACK:
            black_region_length += 1
            white_region_length = 0
            if white_pixel_count > white_threshold and pixel_count_end_of_first_painting > 0:
                if black_region_length > MINIMUM_BLACK_WIDTH*width:
                    current_region = colorRegion.WHITE
                    number_of_white_predominant_regions += 1
                    if pixel_count_end_of_first_painting > 0:
                        pixel_count_start_of_second_painting = i
                else:
                    continue
            elif white_pixel_count > white_threshold:
                current_region = colorRegion.WHITE
                number_of_white_predominant_regions += 1
                if pixel_count_end_of_first_painting > 0:
                    pixel_count_start_of_second_painting = i
        elif current_region == colorRegion.WHITE:
            black_region_length = 0
            white_region_length += 1
            if white_pixel_count < black_threshold:
                if white_region_length > MINIMUM_WHITE_WIDTH*width:
                    if pixel_count_start_of_second_painting <= 0:
                        pixel_count_end_of_first_painting = i
                else:
                    number_of_white_predominant_regions -= 1
                current_region = colorRegion.BLACK

    if current_region == colorRegion.WHITE and white_region_length < MINIMUM_WHITE_WIDTH*width:
        number_of_white_predominant_regions -= 1

    if number_of_white_predominant_regions > 1:
        print(pixel_count_end_of_first_painting,pixel_count_start_of_second_painting)

    print("Count of paintings: " + str(number_of_white_predominant_regions))
    pixel_count_cut = int(pixel_count_end_of_first_painting + ((pixel_count_start_of_second_painting - pixel_count_end_of_first_painting) / 2))
    return number_of_white_predominant_regions, pixel_count_cut

def splitImageInTwo(img, cutPoint):
    print("\nSplitting image at cutPoint " + str(cutPoint))
    img1 = img[:,:cutPoint,:]
    img2 = img[:,cutPoint:,:]
    return [img1, img2]

class colorRegion(Enum):
    BLACK = 0,
    WHITE = 1

def getListOfPaintings(img_list,method):
    output = []
    img_list = [item[0] for item in img_list]
    for img in img_list:
        if method == "MASK":
            mask = computeMaskToCountPaintings(img)
            count, cut = countNumberPaintingsBasedOnMask(mask, img)
        elif method == "EDGES":
            mask = computeEdgesToCountPaintings(img)
            count, cut = countNumberPaintingsBasedOnEdges(mask, img)
        if count == 2:
            print("-- Cut at pixel: ", str(cut))
            output.append(splitImageInTwo(img, cut))
        else:
            output.append([img])
    return output


def main(img_folder,method):
    error_images = []
    for img_path in sorted(glob(os.path.join(img_folder, "*.jpg")))[:]:
        expected = 1
        if any([item in img_path for item in ["00001.jpg","00004.jpg","00009.jpg","00017.jpg","00018.jpg","00025.jpg","00028.jpg"]]):
            expected = 2

        if method == "MASK":
            mask = computeMaskToCountPaintings(img_path)
            count, cut = countNumberPaintingsBasedOnMask(mask, img_path)
        elif method == "EDGES":
            mask = computeEdgesToCountPaintings(img_path)
            count, cut = countNumberPaintingsBasedOnEdges(mask, img_path)

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
    imgs_folder = r"C:\Users\PC\Documents\Roger\Master\M1\Project\Week2\qsd2_w2"
    main(imgs_folder,method="EDGES")

# METHOD MASK
'''Anotate results:
Note that we prefer to predict 2 in one, that missing a picture
For params: 0.6, 0.5 => 00000.jpg bad guess counts 2 instead of 1, 00017.jpg bad guess counts 1 instead of 2. Seems hard to correct both
                        since they have two opposite strategies, In case of 0000.jpg if I push the white threshold to 0.7 can be solved,
                        but the problem of 00017.jpg is that the first of the paintings is wrong
For params: 0.7, 0.5 => Fix 00000.jpg but breaks 00004.jpg, 000018 and 00025 and still has 00017 wrong
For params: 0.65, 0.5 => Fix 000018.jpg and 00004.jpg but breaks back 00000.jpg, and still has 00017 and 00018 wrong

With params 0.6, 0.5 => Change the kernel size from (10,10) (40,40) to a (30,30), (40,40) => Bad guess in 4.jpg, 9.jpg, 28.jpg
'''

# METHOD EDGES
'''Results annotation:
It only fails for image 18 because it has a horizontal structure in the bottom of the image that prevents detecting the division
between the two paintings.
'''