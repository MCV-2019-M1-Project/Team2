# -- IMPORTS -- #
from glob import glob
from paintings_count import getListOfPaintings
from background_removal import BackgroundMask4
from textbox_removal import TextBoxRemoval
from descriptor import SubBlockDescriptor
from searcher import Searcher
from evaluation import EvaluateDescriptors
import numpy as np
import pickle
import cv2
import os

# -- DIRECTORIES -- #
db = '../database'
qs1_w2 = '../qsd1_w2'
qs2_w2 = '../qsd2_w2'
mask_root = '../results/QS2_masks'
res_root = '../results'


def main_qs2():
    # -- GET IMAGES -- #
    folder_path = r"C:\Users\PC\Documents\Roger\Master\M1\Project\Week2\qsd2_w2"
    img_paths = sorted(glob(os.path.join(folder_path,"*.jpg")))
    print("Obtaining list of paintings...")
    img2paintings = getListOfPaintings(folder_path,"EDGES")
    db_images = []
    for db_path in sorted(glob(os.path.join(db,"*.jpg"))):
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
        cv2.imwrite(img_path.replace(".jpg","_fg_mask.png"),np.concatenate([item["mask"] for item in img2paintings_mask[-1]],axis=1))
    print("Done.")

    print("Obtaining textbox masks for each painting...")
    img2paintings_items = []
    img2paintings_bboxs = []
    for ind,img in enumerate(img2paintings_mask):
        print(ind,"of",len(img2paintings_mask))
        img2paintings_items.append([])
        for painting_items in img:
            painting_masked = painting_items["painting"][painting_items["mean_points"]["top"]:painting_items["mean_points"]["bottom"],painting_items["mean_points"]["left"]:painting_items["mean_points"]["right"],:]
            # cv2.imwrite(r"C:\Users\PC\Documents\Roger\Master\M1\Project\Week2\qsd2_w2\test2.png",painting_masked)
            mask, textbox = TextBoxRemoval(painting_masked)
            # cv2.imwrite(r"C:\Users\PC\Documents\Roger\Master\M1\Project\Week2\qsd2_w2\test.png",mask)
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
    with open(os.path.join(folder_path,"final_bboxs.pkl"),"wb") as f:
        pickle.dump(final_bboxs,f)
    print("Done.")

    print("Writing final masks...")
    for img_path,final_mask in zip(img_paths,final_masks):
        cv2.imwrite(img_path.replace(".jpg","_final_mask.png"),final_mask)
    print("Done.")

    print('Obtaining descriptors.')
    # -- DESCRIPTORS -- #
    db_desc = SubBlockDescriptor(db_images,None,flag=False)
    db_desc.compute_descriptors(grid_blocks=[8,8],quantify=[32,8,8],color_space='hsv')
    q2_desc = SubBlockDescriptor(img2paintings,img2paintings_final_mask)
    q2_desc.compute_descriptors(grid_blocks=[8,8],quantify=[32,8,8],color_space='hsv')

    # -- SEARCH -- #
    q2_searcher = Searcher(db_desc.result,q2_desc.result)
    db_desc.clear_memory()
    q2_desc.clear_memory()
    q2_searcher.search(limit=3)

# -- MAIN FOR QSD1_W2 -- #
def main_qs1():

    # -- GET IMAGES -- #
    qimg_paths = sorted(glob(os.path.join(qs1_w2,"*.jpg")))
    query_images = getListOfPaintings(qs1_w2,"EDGES")
    db_images = []
    for db_path in sorted(glob(os.path.join(db,"*.jpg"))):
        db_images.append([cv2.imread(db_path)])

    # -- REMOVE TEXT and GENERATE MASK -- #
    query_mask = []
    query_bbox = []
    for ind,img in enumerate(query_images):
        for paint in img:
            mask, textbox = TextBoxRemoval(paint)
            bbox = [textbox[0][1],textbox[0][0],textbox[1][1],textbox[1][0]]
            query_mask = [mask]
            query_bbox = [bbox]
            cv2.imwrite(os.path.join(res_root,"QS1W2/{0:05d}.png".format(ind)),mask)
    
    # -- SAVE BBOXES -- #
    with open(os.path.join(res_root,"qs1_bbox.pkl"),'wb') as file:
        pickle.dump(query_bbox,file)

    # -- DESCRIPTORS -- #
    db_desc = SubBlockDescriptor(db_images,None,flag=False)
    db_desc.compute_descriptors(grid_blocks=[8,8],quantify=[32,8,8],color_space='hsv')
    q1_desc = SubBlockDescriptor(query_images,query_mask)
    q1_desc.compute_descriptors(grid_blocks=[8,8],quantify=[32,8,8],color_space='hsv')

    # -- SEARCH -- #
    q1_searcher = Searcher(db_desc.result,q1_desc.result)
    db_desc.clear_memory()
    q1_desc.clear_memory()
    q1_searcher.search(limit=3)

    # -- EVALUATE DESCRIPTORS -- #
    q1_eval = EvaluateDescriptors(q1_searcher.result,os.path.join(qs1_w2,"gt_corresps.pkl"))
    q1_searcher.clear_memory()
    q1_eval.compute_mapatk(limit=3)

if __name__ == '__main__':
    main_qs1()
    main_qs2()
