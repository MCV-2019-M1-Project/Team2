# -- IMPORTS -- #
from glob import glob
from paintings_count import getListOfPaintings
from background_removal import BackgroundMask4
from textbox_removal import TextBoxRemoval
from descriptor import SubBlockDescriptor,TransformDescriptor, TextDescriptor
from searcher import Searcher, SearcherCombined, SearcherText
from evaluation import EvaluateDescriptors, EvaluateIoU
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
tests_path = "../tests_folder"
qs1_corresps_path = qs1_w3 + "/gt_corresps.pkl"
qs2_corresps_path = qs2_w3 + "/gt_corresps.pkl"

def get_text():
	text = sorted(glob(dbt+os.sep+'*.txt'))
	db_text = {}
	for k,path in enumerate(text):
		db_text[k] = []
		with open(path,'r') as ff:
			line = ff.readline()
			if not line:
				db_text[k].append([line])
			else:
				line = line.split(',')
				db_text[k].append([line[0][2:-1]])
	return db_text

def main_qs1w3(evaluate=False):
	print("QSD1_W3")
	print("Reading Images...")
	db_text = get_text()
	denoiser = Denoise(qs1_w3)
	db_images = [[cv2.imread(item)] for item in sorted(glob(os.path.join(db,"*.jpg")))]
	print("Denoising Images...")
	denoiser.median_filter(3)
	qs_images = denoiser.tv_bregman(weight=0.01,max_iter=1000,eps=0.001,isotropic=True)
	# cv2.imwrite(r"C:\Users\PC\Documents\Roger\Master\M1\Project\Week3\tests_folder\testdenoise.png",qs_images[0][0])
	# qs_images = [[cv2.imread(item)] for item in sorted(glob(os.path.join(qs1_w3,"*.jpg")))] # No denoising
	print("Done.")

	print("Obtaining textbox masks for each painting...")
	query_mask = []
	query_bbox = []
	for ind,img in enumerate(qs_images):
		print(ind,"of",len(qs_images))
		for paint in img:
			mask, textbox = TextBoxRemoval(paint)
			bbox = [textbox[0][1],textbox[0][0],textbox[1][1],textbox[1][0]]
			query_mask.append([mask])
			query_bbox.append([bbox])
			cv2.imwrite(os.path.join(res_root,'QS1W3','{0:05d}.png'.format(ind)),mask)
			# cv2.imwrite(os.path.join(tests_path,'{0:05d}_mask.png'.format(ind)),mask)
	print("Done.")
	# input("Stop execution...")

	if evaluate:
		eval_iou = EvaluateIoU(query_bbox,os.path.join(qs1_w3,"text_boxes.pkl"))
		eval_iou.compute_iou()
		print("Bbox masks IoU:",eval_iou.score)

	# -- SAVE BBOXES -- #
	print("Writing final bboxs...")
	with open(os.path.join(res_root,"qs1_bbox.pkl"),'wb') as file:
		pickle.dump(query_bbox,file)
	print("Done.")

	# -- DESCRIPTORS -- #
	# -- COLOR -- #
	print('computing color descriptors')
	db_desc_col = SubBlockDescriptor(db_images,None)
	db_desc_col.compute_descriptors()
	qs_desc_col = SubBlockDescriptor(qs_images,query_mask)
	qs_desc_col.compute_descriptors()
	# -- SEARCH -- #
	qs_searcher = Searcher(db_desc_col.result,qs_desc_col.result)
	qs_searcher.search(limit=10)
	if evaluate:
		evaluator = EvaluateDescriptors(qs_searcher.result, qs1_corresps_path)
		map_at_1 = evaluator.compute_mapatk(1)
		map_at_5 = evaluator.compute_mapatk(5)
		print("MAP@1 for color descriptors ", map_at_1)
		print("MAP@5 for color descriptors ", map_at_5)
	print("Done.")

	print("Writing color desc...")
	with open(os.path.join(res_root,"qs1_color_result.pkl"),'wb') as file:
		pickle.dump(qs_searcher.result,file)
	print("Done.")

	# -- TRANSFORM -- #
	print('Obtaining transform descriptors.')
	db_desc_trans = TransformDescriptor(db_images,None,None)
	db_desc_trans.compute_descriptors(transform_type='hog')
	qs_desc_trans = TransformDescriptor(qs_images,query_mask,None)
	qs_desc_trans.compute_descriptors(transform_type='hog')
	# -- SEARCH -- #
	qs_searcher = Searcher(db_desc_trans.result,qs_desc_trans.result)
	qs_searcher.search(limit=10)
	if evaluate:
		evaluator = EvaluateDescriptors(qs_searcher.result, qs1_corresps_path)
		map_at_1 = evaluator.compute_mapatk(1)
		map_at_5 = evaluator.compute_mapatk(5)
		print("MAP@1 for transform descriptors ", map_at_1)
		print("MAP@5 for transform descriptors ", map_at_5)
	print("Done.")

	print("Writing transform desc...")
	with open(os.path.join(res_root,"qs1_transform_result.pkl"),'wb') as file:
		pickle.dump(qs_searcher.result,file)
	print("Done.")

	# -- TEXT -- #
	print('computing text descriptors')
	qs_desc_text = TextDescriptor(qs_images,query_bbox)
	qs_desc_text.compute_descriptors()
	# -- SEARCH -- #
	qs_searcher = SearcherText(db_text,qs_desc_text.result)
	qs_desc_text.clear_memory()
	qs_searcher.search(limit=10)
	if evaluate:
		evaluator = EvaluateDescriptors(qs_searcher.result, qs1_corresps_path)
		map_at_1 = evaluator.compute_mapatk(1)
		map_at_5 = evaluator.compute_mapatk(5)
		print("MAP@1 for text descriptors with levensthein ", map_at_1)
		print("MAP@5 for text descriptors with levensthein ", map_at_5)
	print("Done.")

	print("Writing text desc...")
	with open(os.path.join(res_root,"qs1_text_result.pkl"),'wb') as file:
		pickle.dump(qs_searcher.result,file)
	print("Done.")

	# -- COMBINED-- #
	print('computing combined descriptors without text')
	# -- SEARCH -- #
	qs_searcher = SearcherCombined(db_desc_col.result,qs_desc_col.result,db_desc_trans.result,qs_desc_trans.result)
	db_desc_col.clear_memory()
	qs_desc_col.clear_memory()
	db_desc_trans.clear_memory()
	qs_desc_trans.clear_memory()
	qs_searcher.search(limit=10)
	if evaluate:
		evaluator = EvaluateDescriptors(qs_searcher.result, qs1_corresps_path)
		map_at_1 = evaluator.compute_mapatk(1)
		map_at_5 = evaluator.compute_mapatk(5)
		print("MAP@1 for combined descriptors without text ", map_at_1)
		print("MAP@5 for combined descriptors without text ", map_at_5)
	print("Done.")

	print("Writing combined desc...")
	with open(os.path.join(res_root,"qs1_combined_without_text_result.pkl"),'wb') as file:
		pickle.dump(qs_searcher.result,file)
	print("Done.")

	# -- COMBINED-- #
	print('computing combined descriptors with text')
	# -- SEARCH -- #
	qs_searcher = SearcherCombined(db_desc_col.result,qs_desc_col.result,db_desc_trans.result,qs_desc_trans.result, db_text, qs_desc_text.result, True)
	db_desc_col.clear_memory()
	qs_desc_col.clear_memory()
	db_desc_trans.clear_memory()
	qs_desc_trans.clear_memory()
	qs_searcher.search(limit=10)
	if evaluate:
		evaluator = EvaluateDescriptors(qs_searcher.result, qs1_corresps_path)
		map_at_1 = evaluator.compute_mapatk(1)
		map_at_5 = evaluator.compute_mapatk(5)
		print("MAP@1 for combined descriptors with text ", map_at_1)
		print("MAP@5 for combined descriptors with text ", map_at_5)
	print("Done.")

	print("Writing combined desc...")
	with open(os.path.join(res_root,"qs1_combined_with_text_result.pkl"),'wb') as file:
		pickle.dump(qs_searcher.result,file)
	print("Done.")


def main_qs2w3(evaluate=False):
	# -- GET IMAGES -- #
	print("Denoising Images...")
	folder_path = qs2_w3
	db_text = get_text()
	denoiser = Denoise(folder_path)
	denoiser.median_filter(3)
	qs_images = denoiser.tv_bregman(weight=0.01,max_iter=1000,eps=0.001,isotropic=True)
	print("Done.")
	print("Obtaining list of paintings...")
	img2paintings = getListOfPaintings(qs_images,"EDGES")
	db_images = []
	for db_path in sorted(glob(os.path.join(db,"*.jpg"))):
		db_images.append([cv2.imread(db_path)])
	print("Done.")

	print("Obtaining background masks for each painting...")
	img2paintings_mask = []
	for ind,img in enumerate(img2paintings):
		print(ind,"of",len(img2paintings))
		img2paintings_mask.append([])
		for painting in img:
			mask, mean_points = BackgroundMask4(painting)
			img2paintings_mask[-1].append({"painting":painting,"mask":mask,"mean_points":mean_points})
			"""UNCOMMENT LINE TO PRODUCE THE MASK TO UPLOAD TO THE SERVER"""
		#cv2.imwrite(os.path.join(res_root,"QS2W3","{0:05d}.png".format(ind)),np.concatenate([item["mask"] for item in img2paintings_mask[-1]],axis=1))
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
			bbox_detected = False if np.mean(mask) == 255 else True
			img2paintings_items[-1].append({"fg_mask":painting_items["mask"],
											"mean_points":painting_items["mean_points"],
											"bbox_mask":bbox_mask,
											"bbox":bbox,
											"bbox_detected":bbox_detected})
	print("Done.")

	print("Combining masks in one picture + adapting bboxes...")
	final_masks = []
	img2paintings_final_mask = []
	img2paintings_fg_bboxs = []
	final_bboxs = []
	for ind,img in enumerate(img2paintings_items):
		print(ind,"of",len(img2paintings_items))
		to_concatenate = []
		fg_bboxs = []
		bboxs = []
		for ind2,painting_items in enumerate(img):
			total_mask = painting_items["fg_mask"]
			if painting_items["bbox_detected"]:
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
			fg_bboxs.append(painting_items["mean_points"])
		img2paintings_fg_bboxs.append(fg_bboxs)
		img2paintings_final_mask.append(to_concatenate)
		final_mask = np.concatenate(to_concatenate,axis=1)
		final_masks.append(final_mask)
		final_bboxs.append(bboxs)
	print("Done.")

	if evaluate:
		eval_iou = EvaluateIoU(final_bboxs,os.path.join(qs2_w3,"text_boxes.pkl"))
		eval_iou.compute_iou()
		print("Bbox masks IoU:",eval_iou.score)

	print("Writing final bboxs...")
	with open(os.path.join(res_root,"qs2_bbox.pkl"),"wb") as f:
		pickle.dump(final_bboxs,f)
	print("Done.")

	print("Writing final masks...")
	for ind,final_mask in enumerate(final_masks):
		cv2.imwrite(os.path.join(res_root,"QS2W3","{0:05d}.png".format(ind)),final_mask)
	print("Done.")

	print("Obtaining descriptors.")

	# -- DESCRIPTORS -- #
	# -- COLOR -- #
	print('computing color descriptors')
	db_desc_col = SubBlockDescriptor(db_images,None)
	db_desc_col.compute_descriptors()
	qs_desc_col = SubBlockDescriptor(img2paintings,img2paintings_final_mask)
	qs_desc_col.compute_descriptors()
	# -- SEARCH -- #
	qs_searcher = Searcher(db_desc_col.result,qs_desc_col.result)
	qs_searcher.search(limit=10)
	if evaluate:
		evaluator = EvaluateDescriptors(qs_searcher.result, qs2_corresps_path)
		map_at_1 = evaluator.compute_mapatk(1)
		map_at_5 = evaluator.compute_mapatk(5)
		print("MAP@1 for color descriptors ", map_at_1)
		print("MAP@5 for color descriptors ", map_at_5)
	print("Done.")

	print("Writing color desc...")
	with open(os.path.join(res_root,"qs2_color_result.pkl"),'wb') as file:
		pickle.dump(qs_searcher.result,file)
	print("Done.")

	# -- TRANSFORM -- #
	print('Obtaining transform descriptors.')
	db_desc_trans = TransformDescriptor(db_images,None,None)
	db_desc_trans.compute_descriptors(transform_type='hog')
	qs_desc_trans = TransformDescriptor(img2paintings,img2paintings_final_mask,img2paintings_fg_bboxs)
	qs_desc_trans.compute_descriptors(transform_type='hog')
	# -- SEARCH -- #
	qs_searcher = Searcher(db_desc_trans.result,qs_desc_trans.result)
	qs_searcher.search(limit=10)
	if evaluate:
		evaluator = EvaluateDescriptors(qs_searcher.result, qs2_corresps_path)
		map_at_1 = evaluator.compute_mapatk(1)
		map_at_5 = evaluator.compute_mapatk(5)
		print("MAP@1 for transform descriptors ", map_at_1)
		print("MAP@5 for transform descriptors ", map_at_5)
	print("Done.")

	print("Writing color desc...")
	with open(os.path.join(res_root,"qs2_transform_result.pkl"),'wb') as file:
		pickle.dump(qs_searcher.result,file)
	print("Done.")

	# -- TEXT -- #
	print('computing text descriptors')
	qs_desc_text = TextDescriptor(img2paintings,img2paintings_fg_bboxs)
	qs_desc_text.compute_descriptors()
	# -- SEARCH -- #
	qs_searcher = SearcherText(db_text,qs_desc_text.result)
	qs_desc_text.clear_memory()
	qs_searcher.search(limit=10)
	if evaluate:
		evaluator = EvaluateDescriptors(qs_searcher.result, qs2_corresps_path)
		map_at_1 = evaluator.compute_mapatk(1)
		map_at_5 = evaluator.compute_mapatk(5)
		print("MAP@1 for text descriptors with levenshtein ", map_at_1)
		print("MAP@5 for text descriptors with levenshtein", map_at_5)
	print("Done.")

	print("Writing text desc...")
	with open(os.path.join(res_root,"qs2_text_result.pkl"),'wb') as file:
		pickle.dump(qs_searcher.result,file)
	print("Done.")

	# -- COMBINED-- #
	print('computing combined descriptors without text')
	# -- SEARCH -- #
	qs_searcher = SearcherCombined(db_desc_col.result,qs_desc_col.result,db_desc_trans.result,qs_desc_trans.result, db_text, qs_desc_text.result)
	db_desc_col.clear_memory()
	qs_desc_col.clear_memory()
	db_desc_trans.clear_memory()
	qs_desc_trans.clear_memory()
	qs_searcher.search(limit=10)
	if evaluate:
		evaluator = EvaluateDescriptors(qs_searcher.result, qs2_corresps_path)
		map_at_1 = evaluator.compute_mapatk(1)
		map_at_5 = evaluator.compute_mapatk(5)
		print("MAP@1 for combined descriptors without text ", map_at_1)
		print("MAP@5 for combined descriptors without text ", map_at_5)
	print("Done.")

	print("Writing combined desc...")
	with open(os.path.join(res_root,"qs2_combined_without_text_result.pkl"),'wb') as file:
		pickle.dump(qs_searcher.result,file)
	print("Done.")

	# -- COMBINED-- #
	print('computing combined descriptors with text')
	# -- SEARCH -- #
	print(db_text)
	print(qs_desc_text.result)
	qs_searcher = SearcherCombined(db_desc_col.result,qs_desc_col.result,db_desc_trans.result,qs_desc_trans.result, db_text, qs_desc_text.result, True)
	db_desc_col.clear_memory()
	qs_desc_col.clear_memory()
	db_desc_trans.clear_memory()
	qs_desc_trans.clear_memory()
	qs_searcher.search(limit=10)
	if evaluate:
		evaluator = EvaluateDescriptors(qs_searcher.result, qs2_corresps_path)
		map_at_1 = evaluator.compute_mapatk(1)
		map_at_5 = evaluator.compute_mapatk(5)
		print("MAP@1 for combined descriptors with text ", map_at_1)
		print("MAP@5 for combined descriptors with text ", map_at_5)
	print("Done.")

	print("Writing combined desc...")
	with open(os.path.join(res_root,"qs2_combined_with_text_result.pkl"),'wb') as file:
		pickle.dump(qs_searcher.result,file)
	print("Done.")

if __name__ == "__main__":
	#main_qs1w3(True)
	main_qs2w3(True)

'''
QS1 Results
Color descriptors => MAP@1 = 0.63333 MAP@5 = 0.70944
Transform descriptors => MAP@1 = 0.9 MAP@5 = 0.925
Text descriptors => MAP@1 = 0.366 MAP@5 = 0.4911
Combined without text  => MAP@1 = 0.9 MAP@5 = 0.925
Combined with text  => MAP@1 = 0.933333 MAP@5 = 0.95

QS2 Results
Color descriptors => MAP@1 = 0.533333 MAP@5 = 0.565
Transform descriptors => MAP@1 = 0.616 MAP@5 = 0.629
Text descriptors => MAP@1 = 0.133 MAP@5 = 0.2038
Combined without text  => MAP@1 = 0.616 MAP@5 = 0.629
Combined with text  => MAP@1 = 0.716 MAP@5 = 0.729
'''
