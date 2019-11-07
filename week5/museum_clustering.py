#  -- IMPORTS -- #
from glob import glob
from transform_descriptor import TransformDescriptor
from subblock_descriptor import SubBlockDescriptor
import cv2
import pickle
import os
import time
from sklearn.cluster import KMeans
import numpy as np
import shutil
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial import distance

# -- DIRECTORIES -- #
db_path = "../bbdd"
res_root = "../results"
cluster_root = res_root+os.sep+'cluster'
NUM_CLUSTERS = 10


def chi2_distance(x, y, eps=1e-10):
	"""FUNCTION::CHI2_DISTANCE:
		>- Returns: The chi squared distance between two arrays.
		Works well with histograms."""
	return np.sum((np.power(np.subtract(x, y), 2) / (np.add(x, y) + eps)), axis=-1)


def main(descriptor='texture'):
	print("\nLoading bbdd images...")
	start = time.time()
	bbdd_images_files = sorted(glob(os.path.join(db_path, "*.jpg"))[:])
	bbdd_images = [[cv2.imread(item)] for item in bbdd_images_files]
	bbdd_images = [[cv2.resize(item[0], (1000, 1000))] for item in bbdd_images]

	print("Done. Time: " + str(time.time() - start))

	## COMPUTING DESCRIPTORS
	start = time.time()
	print("\nComputing descriptors for bbdd images...")
	if descriptor == 'texture':
		bbdd_descriptor = TransformDescriptor(bbdd_images, None, None)
		bbdd_results = bbdd_descriptor.compute_descriptors(transform_type='hog')
		cluster_folder = cluster_root + "_texture"
	elif descriptor == 'color':
		bbdd_descriptor = SubBlockDescriptor(bbdd_images, None)
		bbdd_results = bbdd_descriptor.compute_descriptors()
		cluster_folder = cluster_root + "_color"
	elif descriptor == 'combined':
		texture_descriptor = TransformDescriptor(bbdd_images, None, None)
		texture_results = texture_descriptor.compute_descriptors(transform_type='hog')
		color_descriptor = SubBlockDescriptor(bbdd_images, None)
		color_results = color_descriptor.compute_descriptors()
		bbdd_results = {}
		for key in texture_results.keys():
			texture_results[key][0].extend(color_results[key][0])
			# texture_results[key].extend(color_results[key])
			bbdd_results[key] = texture_results[key]
		cluster_folder = cluster_root + "_combined"
	print("Done. Time: " + str(time.time() - start))

	start = time.time()
	X = np.array([np.array(result).reshape(-1) for (k, result) in bbdd_results.items()])
	print("Computing Kmeans clustering...")
	kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0).fit(X)
	print("Done")

	print('Creating Directories')
	if not os.path.isdir(cluster_folder):
		os.mkdir(cluster_folder)
	for (idx, cluster_value) in enumerate(kmeans.labels_):
		file = bbdd_images_files[idx]
		folder = cluster_folder+os.sep+str(cluster_value+1)
		if not os.path.isdir(folder):
			os.mkdir(folder)
		shutil.copy2(file, folder)
	print("Done")

	centers = np.array(kmeans.cluster_centers_)
	clusters = kmeans.labels_.tolist()

	closest_data = {}
	for i in range(NUM_CLUSTERS):
		center_vec = centers[i].reshape(1, -1)
		data_idx_within_i_cluster = [idx for idx, clu_num in enumerate(clusters) if clu_num == i]
		cluster_size = len(data_idx_within_i_cluster)
		if cluster_size > 0:
			X_cluster = np.zeros((len(data_idx_within_i_cluster), centers.shape[1]))
			distances = {}
			for row_num, data_idx in enumerate(data_idx_within_i_cluster):
				one_row = X[data_idx]
				X_cluster[row_num] = one_row
				distances[data_idx] = distance.euclidean(one_row, center_vec[0]) # np.linalg.norm(one_row - center_vec[0])

			closest, _ = pairwise_distances_argmin_min(center_vec, X_cluster)
			closest_to_find = min(5, cluster_size)
			closest_id_in_X_cluster = closest[:closest_to_find]
			closest_data_row_num = np.array(data_idx_within_i_cluster)[closest_id_in_X_cluster.astype(int)]
			data_id = np.array(bbdd_images_files)[closest_data_row_num.astype(int)]
			data2_id = sorted(distances, key=distances.get)[:closest_to_find]
			print("Closest images to cluster center for cluster " + str(i + 1))
			print(data2_id)
			closest_data[i] = data2_id

	print('Save closest data')
	with open(cluster_folder+os.sep+'closest.pkl','wb') as ff:
		pickle.dump(closest_data,ff)
	print("Done")
	
	print("Done. Time: " + str(time.time() - start))


if __name__ == "__main__":
	main('texture')
