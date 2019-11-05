#  -- IMPORTS -- #
from glob import glob
from transform_descriptor import TransformDescriptor
import cv2
import os
import time
from sklearn.cluster import KMeans
import numpy as np

# -- DIRECTORIES -- #
db_path = "../bbdd"
res_root = "../results"


def main():
    start = time.time()

    print("\nLoading bbdd images...")
    start = time.time()
    bbdd_images = [[cv2.imread(item)] for item in sorted(glob(os.path.join(db_path, "*.jpg"))[:])]
    bbdd_images = [[cv2.resize(item[0], (1000, 1000))] for item in bbdd_images]
    print(len(bbdd_images))
    print("Done. Time: " + str(time.time() - start))

    ## COMPUTING DESCRIPTORS
    print("\nComputing descriptors for bbdd images...")
    start = time.time()
    bbdd_descriptor = TransformDescriptor(bbdd_images, None, None)
    bbdd_results = bbdd_descriptor.compute_descriptors(transform_type='hog')
    for k,v in bbdd_results.items():
        x = np.array(v)
        print(x.shape)
    X = np.array([np.array(result) for (k, result) in bbdd_results.items()])
    print(X.shape)
    kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
    print(kmeans.labels_)

    #print(type(bbdd_results))
    #print(len(bbdd_results))
    #print(bbdd_results[0].len)
    #print(len(bbdd_results[1]))
    #print(type(bbdd_results[0]))
    #print(type(bbdd_results[1]))
    #print(bbdd_results[0])
    #print(bbdd_results[1])
    print("Done. Time: " + str(time.time() - start))


if __name__ == "__main__":
    main()
