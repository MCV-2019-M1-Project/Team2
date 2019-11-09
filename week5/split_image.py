# -- IMPORTS -- #
from paintings_count import computeEdgesToCountPaintings, countNumberPaintingsBasedOnEdges
from paintings_count import splitImage
import numpy as np
import cv2

class SplitImages():
    """CLASS::SplitImages:
        >- Splits the images to get the paintings separately."""
    def __init__(self,img_list):
        self.img_list = img_list
        self.output = []

    def get_paintings(self):
        for k,img in enumerate(self.img_list):
            mask = computeEdgesToCountPaintings(img)
            cut_points, display = countNumberPaintingsBasedOnEdges(mask)
            if len(cut_points) > 1:
                print("-- Cut at pixel/s: ", cut_points, "Display", display)
                self.output.append(splitImage(img, cut_points, display))
            else:
                self.output.append([img])
            for s,split in enumerate(self.output[-1]):
                cv2.imwrite('../results/Split/{0:02}_{1}.png'.format(k,s),split)
        return self.output