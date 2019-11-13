import os
from enum import Enum
from glob import glob
import cv2
import numpy as np

# Parameters EDGES method
MINIMUM_BLACK_LENGTH = -1 # -1 FOR NO USE
MINIMUM_WHITE_LENGTH = 0.1
MINIMUM_WHITE_COUNT = 0.02

def computeEdgesToCountPaintings(img):
    # Read image and resize
    original_size = img.shape[:2]
    img = cv2.resize(img,(1000,1000))

    # Obtain gray level image
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    gray = gray[:,:,0]
    #cv2.imwrite(r"C:\Users\PC\Documents\Roger\Master\M1\Project\Week3\tests_folder\testpaintcount.png",gray)

    # Obtain edges through Canny algorithm
    edges = cv2.Canny(gray,50,150,apertureSize=3)

    # Dilate edges and resize
    kernel = np.ones((3,3),np.uint8)
    edges = cv2.dilate(edges,kernel,iterations=1)
    edges = cv2.resize(edges,(original_size[1],original_size[0]))

    # cv2.imwrite(r"C:\Users\PC\Documents\Roger\Master\M1\Project\Week5\qsd1_w5_denoised_rotated\00004_edges.png",edges)

    return edges

def countNumberPaintingsBasedOnEdges(edges):
    print("\nCounting number of paintings")
    height = edges.shape[0]
    width = edges.shape[1]

    ## --- COUNT HORIZONTALLY ---
    # white_threshold = 0
    # black_threshold = 1
    white_threshold = int(height*MINIMUM_WHITE_COUNT)
    black_threshold = int(height*MINIMUM_WHITE_COUNT)

    current_region = colorRegion.BLACK
    black_region_length = 0
    white_region_length = 0
    paintings_col_inds = []

    for col_ind in range(width):
        col_pixels = edges[:,col_ind]
        # white_pixel_count = np.count_nonzero(col_pixels)
        first_nonzero = next((ind for ind,item in enumerate(col_pixels) if item != 0), None)
        last_nonzero = next((ind for ind,item in enumerate(col_pixels[::-1]) if item != 0), None)
        if first_nonzero is not None: #This means, by force, that last_nonzero is also not None
            last_nonzero = height - last_nonzero
            white_pixel_count = last_nonzero - first_nonzero + 1
        else:
            white_pixel_count = 0

        # print(white_pixel_count)

        if current_region == colorRegion.BLACK:
            black_region_length += 1
            white_region_length = 0
            if white_pixel_count > white_threshold:
                current_region = colorRegion.WHITE
                paintings_col_inds.append([])
                paintings_col_inds[-1].append(col_ind)

        elif current_region == colorRegion.WHITE:
            white_region_length += 1
            black_region_length = 0
            if white_pixel_count < black_threshold:
                current_region = colorRegion.BLACK
                paintings_col_inds[-1].append(col_ind)

    adjusted_paintings_col_inds = []
    for item in paintings_col_inds:
        if len(item) == 1:
            item.append(width)
        if len(item) != 2:
            raise Exception("** UNEXPECTED RESULT IN PAINTINGS COUNT **")
        if item[1]-item[0] >= MINIMUM_WHITE_LENGTH*width:
            adjusted_paintings_col_inds.append(item)

    readjusted_paintings_col_inds = []
    consider_ind = True
    for ind,item in enumerate(adjusted_paintings_col_inds):
        if consider_ind:
            if ind < len(adjusted_paintings_col_inds)-1:
                if adjusted_paintings_col_inds[ind+1][0]-adjusted_paintings_col_inds[ind][1] < MINIMUM_BLACK_LENGTH*width:
                    readjusted_paintings_col_inds.append([adjusted_paintings_col_inds[ind][0],adjusted_paintings_col_inds[ind+1][1]])
                    consider_ind = False
                else:
                    readjusted_paintings_col_inds.append(item)
            else:
                readjusted_paintings_col_inds.append(item)

        else:
            consider_ind = True

    print("paintings_col_inds",paintings_col_inds)
    print("adjusted_paintings_col_inds",adjusted_paintings_col_inds)
    print("readjusted_paintings_col_inds",readjusted_paintings_col_inds)

    cut_points = [0]
    for ind,_ in enumerate(readjusted_paintings_col_inds):
        if ind < len(readjusted_paintings_col_inds)-1:
            cut_point = int((readjusted_paintings_col_inds[ind][1]+readjusted_paintings_col_inds[ind+1][0])/2)
            cut_points.append(cut_point)
    cut_points.append(width)

    final_horizontal_cut_points = []
    for ind,cut_point in enumerate(cut_points):
        if ind < len(cut_points)-1:
            final_horizontal_cut_points.append([cut_points[ind],cut_points[ind+1]])

    ## --- COUNT VERTICALLY  ---
    # white_threshold = 0
    # black_threshold = 1
    white_threshold = int(width*MINIMUM_WHITE_COUNT)
    black_threshold = int(width*MINIMUM_WHITE_COUNT)

    current_region = colorRegion.BLACK
    black_region_length = 0
    white_region_length = 0
    paintings_row_inds = []

    for row_ind in range(height):
        row_pixels = edges[row_ind,:]
        # white_pixel_count = np.count_nonzero(row_pixels)
        first_nonzero = next((ind for ind,item in enumerate(row_pixels) if item != 0), None)
        last_nonzero = next((ind for ind,item in enumerate(row_pixels[::-1]) if item != 0), None)
        if first_nonzero is not None: #This means, by force, that last_nonzero is also not None
            last_nonzero = width - last_nonzero
            white_pixel_count = last_nonzero - first_nonzero + 1
        else:
            white_pixel_count = 0

        if current_region == colorRegion.BLACK:
            black_region_length += 1
            white_region_length = 0
            if white_pixel_count > white_threshold:
                current_region = colorRegion.WHITE
                paintings_row_inds.append([])
                paintings_row_inds[-1].append(row_ind)

        elif current_region == colorRegion.WHITE:
            white_region_length += 1
            black_region_length = 0
            if white_pixel_count < black_threshold:
                current_region = colorRegion.BLACK
                paintings_row_inds[-1].append(row_ind)

    adjusted_paintings_row_inds = []
    for item in paintings_row_inds:
        if len(item) == 1:
            item.append(height)
        if len(item) != 2:
            raise Exception("** UNEXPECTED RESULT IN PAINTINGS COUNT **")
        if item[1]-item[0] >= MINIMUM_WHITE_LENGTH*height:
            adjusted_paintings_row_inds.append(item)

    readjusted_paintings_row_inds = []
    consider_ind = True
    for ind,item in enumerate(adjusted_paintings_row_inds):
        if consider_ind:
            if ind < len(adjusted_paintings_row_inds)-1:
                if adjusted_paintings_row_inds[ind+1][0]-adjusted_paintings_row_inds[ind][1] < MINIMUM_BLACK_LENGTH*height:
                    readjusted_paintings_row_inds.append([adjusted_paintings_row_inds[ind][0],adjusted_paintings_row_inds[ind+1][1]])
                    consider_ind = False
                else:
                    readjusted_paintings_row_inds.append(item)
            else:
                readjusted_paintings_row_inds.append(item)
        else:
            consider_ind = True

    print("paintings_row_inds",paintings_row_inds)
    print("adjusted_paintings_row_inds",adjusted_paintings_row_inds)
    print("readjusted_paintings_row_inds",readjusted_paintings_row_inds)

    cut_points = [0]
    for ind,_ in enumerate(readjusted_paintings_row_inds):
        if ind < len(readjusted_paintings_row_inds)-1:
            cut_point = int((readjusted_paintings_row_inds[ind][1]+readjusted_paintings_row_inds[ind+1][0])/2)
            cut_points.append(cut_point)
    cut_points.append(height)

    final_vertical_cut_points = []
    for ind,cut_point in enumerate(cut_points):
        if ind < len(cut_points)-1:
            final_vertical_cut_points.append([cut_points[ind],cut_points[ind+1]])

    ### --- CHOOSE HORIZONTAL OR VERTICAL CUT POINTS ---
    print("\tNum horizontal cut points:", len(final_horizontal_cut_points), "Num vertical cut points:", len(final_vertical_cut_points))
    if len(final_horizontal_cut_points) >= len(final_vertical_cut_points):
        final_cut_points = final_horizontal_cut_points
        display = "horizontal"
    else:
        final_cut_points = final_vertical_cut_points
        display = "vertical"

    return final_cut_points, display

def splitImage(img, cut_points, display):
    print("\tSplitting image at cutPoints",cut_points,"Display:",display)
    imgs = []
    for item in cut_points:
        if display == "horizontal":
            imgs.append(img[:,item[0]:item[1],:])
        elif display == "vertical":
            imgs.append(img[item[0]:item[1],:,:])
    return imgs

class colorRegion(Enum):
    BLACK = 0,
    WHITE = 1

class SplitImages():
    """CLASS::SplitImages:
        >- Splits the images to get the paintings separately."""
    def __init__(self,img_list):
        self.img_list = img_list
        self.output = []
        self.displays = []

    def get_paintings(self):
        for k,img in enumerate(self.img_list):
            mask = computeEdgesToCountPaintings(img)
            cut_points, display = countNumberPaintingsBasedOnEdges(mask)
            self.displays.append(display)
            if len(cut_points) > 1:
                print("-- Cut at pixel/s: ", cut_points, "Display", display)
                self.output.append(splitImage(img, cut_points, display))
            else:
                self.output.append([img])
            for s,split in enumerate(self.output[-1]):
                cv2.imwrite('../results/Split/{0:02}_{1}.png'.format(k,s),split)
        return self.output, self.displays

def main(img_folder,method):
    error_images = []
    for img_path in sorted(glob(os.path.join(img_folder, "*.jpg")))[:]:
        expected = 1
        if "qsd2_w2" in img_folder:
            if any([item in img_path for item in ["00001","00004","00009","00017","00018","00025","00028"]]):
                expected = 2
        elif "qsd2_w3" in img_folder:
            if any([item in img_path for item in ["00003","00017","00018","00019","00023","00025"]]):
                expected = 2
        elif "qsd1_w4" in img_folder:
            if any([item in img_path for item in ["00000","00003","00005","00007","00009","00017","00018","00023","00027"]]):
                expected = 2
            if any([item in img_path for item in ["00030"]]):
                expected = 3
        elif "qsd1_w5" in img_folder:
            if any([item in img_path for item in ["00000","00001","00002","00003","00004","00005","00008","00014","00017","00026","00027","00029"]]):
                expected = 2
            if any([item in img_path for item in ["00007","00010","00012","00015","00016","00018","00019","00023"]]):
                expected = 3

        img = cv2.imread(img_path)

        if method == "EDGES":
            mask = computeEdgesToCountPaintings(img)
            cut_points, display = countNumberPaintingsBasedOnEdges(mask)

        if len(cut_points) == expected:
            print("-- Good guess for image: ", img_path)
        else:
            error_images.append(img_path)
            print("-- Error for image " + img_path + ", expected: " + str(expected) + ", actual: " + str(len(cut_points)))

        split_images = splitImage(img, cut_points, display)
        total_length = 0
        for ind,split_img in enumerate(split_images):
            cv2.imwrite(img_path.replace("qsd1_w5_denoised_rotated","qsd1_w5_denoised_rotated_split").replace(".jpg","_"+str(ind)+".jpg"),split_img)
            if display == "horizontal":
                total_length += split_img.shape[1]
            else:
                total_length += split_img.shape[0]
        if display == "horizontal":
            if total_length != img.shape[1]:
                print(total_length,img.shape[1])
                raise Exception("** ERROR - WEIRD SHAPES **")
        else:
            if total_length != img.shape[0]:
                print(total_length,img.shape[0])
                raise Exception("** ERROR - WEIRD SHAPES **")

    print("-- Pictures with wrong guesses --")
    print(error_images)


if __name__ == '__main__':
    # imgs_folder = r"C:\Users\PC\Documents\Roger\Master\M1\Project\Week5\qsd1_w4_denoised"
    # imgs_folder = r"C:\Users\PC\Documents\Roger\Master\M1\Project\Week5\qsd2_w3_denoised"
    # imgs_folder = r"C:\Users\PC\Documents\Roger\Master\M1\Project\Week5\qsd2_w2_denoised"
    # imgs_folder = r"C:\Users\PC\Documents\Roger\Master\M1\Project\Week5\qsd1_w5_denoised"
    imgs_folder = r"C:\Users\PC\Documents\Roger\Master\M1\Project\Week5\qsd1_w5_denoised_rotated"
    main(imgs_folder,method="EDGES")

"""
MINIMUM_BLACK_LENGTH = 0.01
MINIMUM_WHITE_LENGTH = 0.1
MINIMUM_WHITE_COUNT = 0.02
    qsd2_w2: 30/30
    qsd2_w3: 30/30
    qsd1_w4: 30/30
    qsd1_w5: 26/30 (2/4 errors no esperats: un per culpa del MINIMUM_BLACK_LENGTH i l'altre ni idea)
"""

"""
MINIMUM_BLACK_LENGTH = -1
MINIMUM_WHITE_LENGTH = 0.1
MINIMUM_WHITE_COUNT = 0.02
    qsd2_w2: 30/30
    qsd2_w3: 30/30
    qsd1_w4: 29/30 (1/1 errors no esperats: culpa del MINIMUM_BLACK_LENGTH)
    qsd1_w5: 27/30 (1/3 errors no esperats: ni idea)
"""
