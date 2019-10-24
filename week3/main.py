# -- IMPORTS -- #
from glob import glob
from paintings_count import getListOfPaintings
from background_removal import BackgroundMask4
from textbox_removal import TextBoxRemoval
from descriptor import SubBlockDescriptor,TransformDescriptor
from searcher import Searcher
from evaluation import EvaluateDescriptors
from noise import Denoise
import numpy as np
import pickle
import cv2
import os

# -- DIRECTORIES -- #
db = "../bbdd"
dbt = "../bbdd_text"
qs1_w3 = "../qsd1_w3"
qs2_w3 = "../qsd2_w3"
res_root = "../results"
masks = "../results/QS1W3"


def main_qs1w3():

	print("QSD1_W3")
	print("Reading Images...")
	denoiser = Denoise(qs1_w3)
	db_images = [[cv2.imread(item)] for item in sorted(glob(os.path.join(db,"*.jpg")))]
	print("Denoising Images...")
	qs_images = denoiser.tv_bregman(weight=0.01,max_iter=1000,eps=0.001,isotropic=True)
	query_mask = [[cv2.imread(item,0)] for item in sorted(glob(os.path.join(masks,"*.png")))]
	print("Done.")

	#print("Obtaining textbox masks for each painting...")
	#query_mask = []
	#query_bbox = []
	#for ind,img in enumerate(qs_images):
	#	print(ind,"of",len(qs_images))
	#	for paint in img:
	#		mask, textbox = TextBoxRemoval(paint)
	#		bbox = [textbox[0][1],textbox[0][0],textbox[1][1],textbox[1][0]]
	#		query_mask.append([mask])
	#		query_bbox.append([bbox])
	#		cv2.imwrite(res_root+os.sep+'QS1W3/{0:05d}.png'.format(ind),mask)
	#print("Done.")

	# -- SAVE BBOXES -- #
	#print("Writing final bboxs...")
	#with open(os.path.join(res_root,"qs1_bbox.pkl"),'wb') as file:
	#	pickle.dump(query_bbox,file)
	#print("Done.")

	# -- DESCRIPTORS -- #
	print('Obtaining descriptors.')
	db_desc = TransformDescriptor(db_images,None,flag=False)
	db_desc.compute_descriptors(transform_type='lbp')
	q1_desc = TransformDescriptor(qs_images,query_mask)
	q1_desc.compute_descriptors(transform_type='lbp')

	# -- SEARCH -- #
	q1_searcher = Searcher(db_desc.result,q1_desc.result)
	db_desc.clear_memory()
	q1_desc.clear_memory()
	q1_searcher.search(limit=3)
	print("Done.")

	q1_eval = EvaluateDescriptors(q1_searcher.result,qs1_w3+os.sep+'gt_corresps.pkl')
	q1_eval.compute_mapatk(limit=1)
	print('DESC MAP1: ['+str(q1_eval.score)+']')
	q1_eval.compute_mapatk(limit=3)
	print('DESC MAP3: ['+str(q1_eval.score)+']')


def main_qs2w3():
	# -- GET IMAGES -- #
	print("Denoising Images...")
	denoiser = Denoise(qs2_w3)
	folder_path = qs2_w3
	img_paths = denoiser.tv_bregman(weight=0.01,max_iter=1000,eps=0.001,isotropic=True)
	print("Done.")
	print("Obtaining list of paintings...")
	img2paintings = getListOfPaintings(folder_path,"EDGES")
	db_images = []
	for db_path in sorted(glob(db+os.sep+"*.jpg")):
		db_images.append([cv2.imread(db_path)])
	print("Done.")

	print("Obtaining background masks for each painting...")
	img2paintings_mask = []
	for ind,(img_path,img) in enumerate(zip(img_paths,img2paintings)):
		print(ind,"of",len(img2paintings))
		img2paintings_mask.append([])
		for painting in img:
			mask, mean_points = BackgroundMask4(painting)
			img2paintings_mask[-1].append({"painting":painting,"mask":mask,"mean_points":mean_points})
		#cv2.imwrite(res_root+os.sep+"QS2W3/{0:05d}.png".format(ind),np.concatenate([item["mask"] for item in img2paintings_mask[-1]],axis=1))
	print("Done.")

	print("Obtaining textbox masks for each painting...")
	img2paintings_items = []
	img2paintings_bboxs = []
	for ind,img in enumerate(img2paintings_mask):
		print(ind,"of",len(img2paintings_mask))
		img2paintings_items.append([])
		for painting_items in img:
			painting_masked = painting_items["painting"][painting_items["mean_points"]["top"]:painting_items["mean_points"]["bottom"],painting_items["mean_points"]["left"]:painting_items["mean_points"]["right"],:]
			mask, textbox = TextBoxRemoval(painting_masked)
			bbox_mask = np.zeros(shape=(painting_items["painting"].shape[0],painting_items["painting"].shape[1]))
			bbox_mask[painting_items["mean_points"]["top"]:painting_items["mean_points"]["bottom"],painting_items["mean_points"]["left"]:painting_items["mean_points"]["right"]] = mask
			bbox = [textbox[0][1],textbox[0][0],textbox[1][1],textbox[1][0]]
			bbox[1] = bbox[1] + painting_items["mean_points"]["top"]
			bbox[3] = bbox[3] + painting_items["mean_points"]["top"]
			bbox[0] = bbox[0] + painting_items["mean_points"]["left"]
			bbox[2] = bbox[2] + painting_items["mean_points"]["left"]
			img2paintings_items[-1].append({"fg_mask":painting_items["mask"],
											"bbox_mask":bbox_mask,
											"bbox":bbox})
	print("Done.")

	print("Combining masks in one picture + adapting bboxes...")
	final_masks = []
	img2paintings_final_mask = []
	final_bboxs = []
	for ind,img in enumerate(img2paintings_items):
		print(ind,"of",len(img2paintings_items))
		to_concatenate = []
		bboxs = []
		for ind2,painting_items in enumerate(img):
			total_mask = painting_items["fg_mask"]
			total_mask[painting_items["bbox"][1]:painting_items["bbox"][3],painting_items["bbox"][0]:painting_items["bbox"][2]] = 0
			to_concatenate.append(total_mask)
			if ind2 == 0:
				bboxs.append(painting_items["bbox"])
			else:
				missing_size = 0
				for item in to_concatenate[:-1]:
					missing_size += item.shape[1]
				bbox = painting_items["bbox"]
				bbox[0] += missing_size
				bbox[2] += missing_size
				bboxs.append(bbox)
		img2paintings_final_mask.append(to_concatenate)
		final_mask = np.concatenate(to_concatenate,axis=1)
		final_masks.append(final_mask)
		final_bboxs.append(bboxs)
	print("Done.")

	print("Writing final bboxs...")
	with open(res_root+os.sep+"qs2_bbox.pkl","wb") as f:
		pickle.dump(final_bboxs,f)
	print("Done.")

	print("Writing final masks...")
	for ind,final_mask in enumerate(final_masks):
		cv2.imwrite(res_root+os.sep+"QS2W3/{0:05d}.png".format(ind),final_mask)
	print("Done.")

	print("Obtaining descriptors.")

	 # -- DESCRIPTORS -- #
	print("Computing descriptors for database images...")
	# db_desc = SubBlockDescriptor(db_images,None,flag=False)
	# db_desc.compute_descriptors(grid_blocks=[8,8],quantify=[32,8,8],color_space="hsv")
	db_desc = TransformDescriptor(db_images,None,flag=False)
	db_desc.compute_descriptors(transform_type="hog")
	print("Done.")
	print("Computing descriptors for query images...")
	# qs_desc = SubBlockDescriptor(qs_images,None,flag=False)
	# qs_desc.compute_descriptors(grid_blocks=[8,8],quantify=[32,8,8],color_space="hsv")
	qs_desc = TransformDescriptor(img2paintings,img2paintings_final_mask,flag=True)
	qs_desc.compute_descriptors(transform_type="hog")
	print("Done.")

	# -- SEARCH -- #
	print("Searching...")
	qs_searcher = Searcher(db_desc.result,qs_desc.result)
	# qs_searcher = SearcherMultiprocess(db_desc.result,qs_desc.result,num_cores=3)
	db_desc.clear_memory()
	qs_desc.clear_memory()
	qs_searcher.search(limit=3)
	print("Done.")

	# -- EVALUATE -- #
	qs_desc_eval = EvaluateDescriptors(qs_searcher.result,qs2_w3+os.sep+'gt_corresps.pkl')
	qs_desc_eval.compute_mapatk(limit=1)
	print('DESC MAP1: ['+str(qs_desc_eval.score)+']')
	qs_desc_eval.compute_mapatk(limit=3)
	print('DESC MAP3: ['+str(qs_desc_eval.score)+']')

if __name__ == "__main__":
	main_qs1w3()
