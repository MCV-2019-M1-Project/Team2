#  -- IMPORTS -- #
from glob import glob
from transform_descriptor import TransformDescriptor
import cv2
import os
import time
from sklearn.cluster import KMeans
import numpy as np
import shutil
from sklearn.metrics import pairwise_distances_argmin_min

# -- DIRECTORIES -- #
db_path = "../bbdd"
res_root = "../results"
cluster_root = "../cluster"
NUM_CLUSTERS = 10


def main():
    start = time.time()

    print("\nLoading bbdd images...")
    start = time.time()
    bbdd_images_files = sorted(glob(os.path.join(db_path, "*.jpg"))[:])
    bbdd_images = [[cv2.imread(item)] for item in bbdd_images_files]
    bbdd_images = [[cv2.resize(item[0], (1000, 1000))] for item in bbdd_images]
    print(len(bbdd_images))
    print("Done. Time: " + str(time.time() - start))

    ## COMPUTING DESCRIPTORS
    print("\nComputing descriptors for bbdd images...")
    start = time.time()
    bbdd_descriptor = TransformDescriptor(bbdd_images, None, None)
    bbdd_results = bbdd_descriptor.compute_descriptors(transform_type='hog')
    X = np.array([np.array(result).reshape(-1) for (k, result) in bbdd_results.items()])
    print(X.shape)
    print("\nComputing Kmeans clustering...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0).fit(X)
    print(kmeans.labels_)

    for (idx, cluster_value) in enumerate(kmeans.labels_):
        file = bbdd_images_files[idx]
        shutil.copy2(file, cluster_root + "/" + str(cluster_value + 1))

    centers = np.array(kmeans.cluster_centers_)
    clusters = kmeans.labels_.tolist()

    closest_data = []
    for i in range(NUM_CLUSTERS):
        print("i " + str(i))
        center_vec = centers[i].reshape(1, -1)
        data_idx_within_i_cluster = [idx for idx, clu_num in enumerate(clusters) if clu_num == i]
        cluster_size = len(data_idx_within_i_cluster)
        if cluster_size > 0:
            print("a " + str(len(data_idx_within_i_cluster)))
            print("b " + str(centers.shape))
            X_cluster = np.zeros((len(data_idx_within_i_cluster), centers.shape[1]))
            print("c " + str(X_cluster.shape))
            for row_num, data_idx in enumerate(data_idx_within_i_cluster):
                one_row = X[data_idx]
                X_cluster[row_num] = one_row

            closest, _ = pairwise_distances_argmin_min(center_vec, X_cluster)
            closest_to_find = min(5, cluster_size)
            closest_idx_in_one_cluster_tf_matrix = closest[0]
            closest_data_row_num = data_idx_within_i_cluster[closest_idx_in_one_cluster_tf_matrix]
            data_id = bbdd_images_files[closest_data_row_num]
            print("Closest images to cluster center for cluster " + str(i))
            print(data_id)
            closest_data.append(data_id)

    print(closest_data)

    print("Done. Time: " + str(time.time() - start))


if __name__ == "__main__":
    main()
