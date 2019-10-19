# -- IMPORTS -- #
from glob import glob
from paintings_count import getListOfPaintings
from background_removal import BackgroundMask4
from textbox_removal import TextBoxRemoval
from descriptor import SubBlockDescriptor
from searcher import Searcher
import numpy as np
import pickle
import cv2
import os

# -- DIRECTORIES -- #
db = '../database'
qs1_w2 = '../qsd1_w2'
qs2_w2 = '../qsd2_w2'
res_root = '../results'


def main_qs2():
    # -- GET IMAGES -- #
    folder_path = qs2_w2
    img_paths = sorted(glob(folder_path+os.sep+'*.jpg'))
    print("Obtaining list of paintings...")
    img2paintings = getListOfPaintings(folder_path,"EDGES")
    db_images = []
    for db_path in sorted(glob(db+os.sep+'*.jpg')):
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
        #cv2.imwrite(res_root+os.sep+'QS2W2/{0:05d}.png'.format(ind),np.concatenate([item["mask"] for item in img2paintings_mask[-1]],axis=1))
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
    with open(res_root+os.sep+'qs2_bbox.pkl',"wb") as f:
        pickle.dump(final_bboxs,f)
    print("Done.")

    print("Writing final masks...")
    for ind,final_mask in enumerate(final_masks):
        cv2.imwrite(res_root+os.sep+'QS2W2/{0:05d}.png'.format(ind),final_mask)
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
    print("Done.")

    # -- SAVE RESULTS -- #
    print('Save results')
    with open(res_root+os.sep+'qs2_result.pkl','wb') as f:
        pickle.dump(q2_searcher.result,f)
    print("Done.")

# -- MAIN FOR QSD1_W2 -- #
def main_qs1():

    # -- GET IMAGES -- #
    print("Obtaining list of paintings...")
    query_images = []
    db_images = []
    for db_path in sorted(glob(db+os.sep+'*.jpg')):
        db_images.append([cv2.imread(db_path)])
    for qimg_paths in sorted(glob(qs1_w2+os.sep+'*.jpg')):
        query_images.append([cv2.imread(qimg_paths)])
    print("Done.")


    # -- REMOVE TEXT and GENERATE MASK -- #
    print("Obtaining textbox masks for each painting...")
    query_mask = []
    query_bbox = []
    for ind,img in enumerate(query_images):
        print(ind,"of",len(query_images))
        for paint in img:
            mask, textbox = TextBoxRemoval(paint)
            bbox = [textbox[0][1],textbox[0][0],textbox[1][1],textbox[1][0]]
            query_mask.append([mask])
            query_bbox.append([bbox])
            cv2.imwrite(res_root+os.sep+'QS1W2/{0:05d}.png'.format(ind),mask)
    print("Done.")

    # -- SAVE BBOXES -- #
    print("Writing final bboxs...")
    with open(os.path.join(res_root,"qs1_bbox.pkl"),'wb') as file:
        pickle.dump(query_bbox,file)
    print("Done.")

    # -- DESCRIPTORS -- #
    print('Obtaining descriptors.')
    db_desc = SubBlockDescriptor(db_images,None,flag=False)
    db_desc.compute_descriptors(grid_blocks=[8,8],quantify=[32,8,8],color_space='hsv')
    q1_desc = SubBlockDescriptor(query_images,query_mask)
    q1_desc.compute_descriptors(grid_blocks=[8,8],quantify=[32,8,8],color_space='hsv')

    # -- SEARCH -- #
    q1_searcher = Searcher(db_desc.result,q1_desc.result)
    db_desc.clear_memory()
    q1_desc.clear_memory()
    q1_searcher.search(limit=3)
    print("Done.")

    # -- SAVE RESULTS -- #
    print('Save results')
    with open(res_root+os.sep+'qs1_result.pkl','wb') as f:
        pickle.dump(q1_searcher.result,f)
    print("Done.")

if __name__ == '__main__':
    main_qs1()
    main_qs2()
