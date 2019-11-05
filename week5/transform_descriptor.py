# -- IMPORTS -- #
from skimage import feature as F
import numpy as np
import cv2


def _compute_hog(img, mask, bbox):
    if mask is not None:
        if bbox is not None:
            img = img[bbox["top"]:bbox["bottom"], bbox["left"]:bbox["right"]]
            mask = mask[bbox["top"]:bbox["bottom"], bbox["left"]:bbox["right"]]
    new_img = cv2.bitwise_and(img, img, mask=mask)
    resized = cv2.resize(new_img, (128, 256), cv2.INTER_AREA)
    winSize = (128, 256)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 5
    derivAperture = 1
    winSigma = 4.0
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    feature = hog.compute(resized, winStride=(8, 8), padding=(8, 8), locations=None).tolist()
    return [item[0] for item in feature]


def _zigzag(a):
    return np.concatenate(
        [np.diagonal(a[::-1, :], i)[::(2 * (i % 2) - 1)] for i in range(1 - a.shape[0], a.shape[0])])


class TransformDescriptor():
    """CLASS::TransformDescriptor:
    >- Class in charge of computing the descriptors for all the images from a directory."""

    def __init__(self, img_list, mask_list=None, bbox_list=None):
        self.img_list = img_list
        if mask_list:
            self.mask_list = mask_list
        else:
            self.mask_list = [[None]] * len(self.img_list)
        if bbox_list:
            self.bbox_list = bbox_list
        else:
            self.bbox_list = [[None]] * len(self.img_list)
        self.result = {}

    def compute_descriptors(self, transform_type='hog', dct_blocks=8, lbp_blocks=15):
        """METHOD::COMPUTE_DESCRIPTORS:
        Computes for each image on the specified data path the correspondant descriptor."""
        self.transform_type = transform_type
        self.lbp_blocks = lbp_blocks
        self.dct_blocks = dct_blocks
        print('--- COMPUTING DESCRIPTORS --- ')
        for k, images in enumerate(self.img_list):
            # print(str(k)+' out of '+str(len(self.img_list)))
            self.result[k] = []
            for i, paint in enumerate(images):
                self.result[k].append(self._compute_features(paint, self.mask_list[k][i], self.bbox_list[k][i]))
        print('--- DONE --- ')
        return self.result

    def clear_memory(self):
        """METHOD::CLEAR_MEMORY:
        >- Deletes the memory allocated that stores data to make it more efficient."""
        self.result = {}

    def _compute_features(self, img, mask, bbox):
        """METHOD::COMPUTE_FEATURES:
        >- Returns the features obtained."""
        if self.transform_type == 'lbp':
            return self._compute_lbp(img, mask, bbox)
        elif self.transform_type == 'dct':
            return self._compute_dct(img, mask)
        elif self.transform_type == 'hog':
            return _compute_hog(img, mask, bbox)

    def _compute_lbp(self, img, mask, bbox):
        features = []
        if mask is not None:
            if bbox is not None:
                img = img[bbox["top"]:bbox["bottom"], bbox["left"]:bbox["right"]]
                mask = mask[bbox["top"]:bbox["bottom"], bbox["left"]:bbox["right"]]
            mask = cv2.resize(mask, (500, 500), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        for i in range(self.lbp_blocks):
            for j in range(self.lbp_blocks):
                new_mask = mask
                if mask is not None:
                    new_mask = mask[int((i / self.lbp_blocks) * mask.shape[0]):int(
                        ((i + 1) / self.lbp_blocks) * mask.shape[0]),
                               int((j / self.lbp_blocks) * mask.shape[1]):int(
                                   ((j + 1) / self.lbp_blocks) * mask.shape[1])]
                new_img = img[int((i / self.lbp_blocks) * img.shape[0]):int(((i + 1) / self.lbp_blocks) * img.shape[0]),
                          int((j / self.lbp_blocks) * img.shape[1]):int(((j + 1) / self.lbp_blocks) * img.shape[1])]
                feature = self._lbp(new_img, new_mask, numPoints=8, radius=2)
                features.extend(feature)
        return features

    def _lbp(self, image, mask, numPoints, radius):
        lbp = F.local_binary_pattern(image, numPoints, radius)
        lbp = np.float32(lbp)
        hist = cv2.calcHist([lbp], [0], mask, [256], [0, 255])
        hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
        hist = hist.flatten()
        return hist

    def _compute_dct(self, img, mask, p=0.05):
        features = []
        num_coeff = int(np.power(512 / self.dct_blocks, 2) * p)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for i in range(self.dct_blocks):
            for j in range(self.dct_blocks):
                new_mask = mask
                if mask is not None:
                    new_mask = mask[int((i / self.dct_blocks) * mask.shape[0]):int(
                        ((i + 1) / self.dct_blocks) * mask.shape[0]),
                               int((j / self.dct_blocks) * mask.shape[1]):int(
                                   ((j + 1) / self.dct_blocks) * mask.shape[1])]
                new_img = img[int((i / self.dct_blocks) * img.shape[0]):int(((i + 1) / self.dct_blocks) * img.shape[0]),
                          int((j / self.dct_blocks) * img.shape[1]):int(((j + 1) / self.dct_blocks) * img.shape[1])]
                transform = cv2.dct(np.float32(new_img) / 255.0)
                coeff = _zigzag(transform)[:num_coeff]
                features.extend(coeff)
        return features
