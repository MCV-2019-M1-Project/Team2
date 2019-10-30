# -- IMPORTS -- #
from glob import glob
from paintings_count import getListOfPaintings
from background_removal import BackgroundMask4
from textbox_removal import TextBoxRemoval
from descriptor import ORBDescriptor
from searcher import Searcher, SearcherCombined, SearcherText
from evaluation import EvaluateDescriptors, EvaluateIoU
from noise import Denoise
import numpy as np
import pickle
import cv2
import os
import time
import bisect

# -- DIRECTORIES -- #
db_path = "../bbdd"
db_text_path = "../bbdd_text"
qs_path = "../qsd1_w4"
res_root = "../results"
tests_path = "../tests_folder"
qs_corresps_path = qs_path + "/gt_corresps.pkl"

def get_text():
    text = sorted(glob(db_text_path + os.sep + '*.txt'))
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

def main_total(evaluate=False):
    # -- GET IMAGES -- #
    print("Denoising Images...")
    folder_path = qs_path
    db_text = get_text()
    denoiser = Denoise(folder_path)
    denoiser.median_filter(3)
    qs_images = denoiser.tv_bregman(weight=0.01,max_iter=1000,eps=0.001,isotropic=True)
    print("Done.")
    print("Obtaining list of paintings...")
    img2paintings = getListOfPaintings(qs_images,"EDGES")
    db_images = []
    for db_path in sorted(glob(os.path.join(db_path, "*.jpg"))):
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
        cv2.imwrite(os.path.join(res_root,"QS2W3","{0:05d}.png".format(ind)),np.concatenate([item["mask"] for item in img2paintings_mask[-1]],axis=1))
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
        eval_iou = EvaluateIoU(final_bboxs, os.path.join(qs_path, "text_boxes.pkl"))
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
    # -- KEYPOINTS -- #
    print('Obtaining keypoint descriptors.')
    db_desc_keypoints = ORBDescriptor(db_images,None,None)
    db_desc_keypoints.compute_descriptors()
    qs_desc_keypoints = ORBDescriptor(img2paintings,img2paintings_final_mask,img2paintings_fg_bboxs)
    qs_desc_keypoints.compute_descriptors()
    # -- SEARCH -- #
    qs_searcher = Searcher(db_desc_keypoints.result,qs_desc_keypoints.result)
    qs_searcher.search(limit=10)
    if evaluate:
        evaluator = EvaluateDescriptors(qs_searcher.result, qs_corresps_path)
        map_at_1 = evaluator.compute_mapatk(1)
        map_at_5 = evaluator.compute_mapatk(5)
        print("MAP@1 for keypoint descriptors ", map_at_1)
        print("MAP@5 for keypoint descriptors ", map_at_5)
    print("Done.")

    print("Writing keypoint desc...")
    with open(os.path.join(res_root,"qs_keypoint_result.pkl"),'wb') as file:
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
        evaluator = EvaluateDescriptors(qs_searcher.result, qs_corresps_path)
        map_at_1 = evaluator.compute_mapatk(1)
        map_at_5 = evaluator.compute_mapatk(5)
        print("MAP@1 for text descriptors with levenshtein ", map_at_1)
        print("MAP@5 for text descriptors with levenshtein", map_at_5)
    print("Done.")

    print("Writing text desc...")
    with open(os.path.join(res_root,"qs_text_result.pkl"),'wb') as file:
        pickle.dump(qs_searcher.result,file)
    print("Done.")

    # -- COMBINED-- #
    print('computing combined descriptors without text')
    # -- SEARCH -- #
    qs_searcher = SearcherCombined(None, None, db_desc_keypoints.result,qs_desc_keypoints.result, db_text, qs_desc_text.result, False)
    db_desc_keypoints.clear_memory()
    qs_desc_keypoints.clear_memory()
    qs_searcher.search(limit=10)
    if evaluate:
        evaluator = EvaluateDescriptors(qs_searcher.result, qs_corresps_path)
        map_at_1 = evaluator.compute_mapatk(1)
        map_at_5 = evaluator.compute_mapatk(5)
        print("MAP@1 for combined descriptors without text ", map_at_1)
        print("MAP@5 for combined descriptors without text ", map_at_5)
    print("Done.")

    print("Writing combined desc...")
    with open(os.path.join(res_root,"qs_combined_without_text_result.pkl"),'wb') as file:
        pickle.dump(qs_searcher.result,file)
    print("Done.")

    # -- COMBINED-- #
    print('computing combined descriptors with text')
    # -- SEARCH -- #
    print(db_text)
    print(qs_desc_text.result)
    qs_searcher = SearcherCombined(None, None, db_desc_keypoints.result, qs_desc_keypoints.result, db_text, qs_desc_text.result, True)
    db_desc_keypoints.clear_memory()
    qs_desc_keypoints.clear_memory()
    qs_searcher.search(limit=10)
    if evaluate:
        evaluator = EvaluateDescriptors(qs_searcher.result, qs_corresps_path)
        map_at_1 = evaluator.compute_mapatk(1)
        map_at_5 = evaluator.compute_mapatk(5)
        print("MAP@1 for combined descriptors with text ", map_at_1)
        print("MAP@5 for combined descriptors with text ", map_at_5)
    print("Done.")

    print("Writing combined desc...")
    with open(os.path.join(res_root,"qs_combined_with_text_result.pkl"),'wb') as file:
        pickle.dump(qs_searcher.result,file)
    print("Done.")

def main_qs1w4():
    global_start = time.time()

    ## LOADING IMAGES
    print("\nLoading query images...")
    start = time.time()
    query_images = [[cv2.imread(item)] for item in sorted(glob(os.path.join(qs_path, "*.jpg"))[:])]
    query_images = [[cv2.resize(item[0],(1000,1000))] for item in query_images]
    print("Done. Time: "+str(time.time()-start))

    print("\nLoading bbdd images...")
    start = time.time()
    bbdd_images = [[cv2.imread(item)] for item in sorted(glob(os.path.join(db_path, "*.jpg"))[:])]
    bbdd_images = [[cv2.resize(item[0],(1000,1000))] for item in bbdd_images]
    print("Done. Time: "+str(time.time()-start))

    ## COMPUTING DESCRIPTORS
    print("\nComputing descriptors for query images...")
    start = time.time()
    # query_descriptors = HarrisDescriptor(query_images, None, None)
    query_descriptor = ORBDescriptor(query_images,None,None)
    query_results = query_descriptor.compute_descriptors()
    print("Done. Time: "+str(time.time()-start))

    print("\nComputing descriptors for bbdd images...")
    start = time.time()
    # bbdd_descriptor = HarrisDescriptor(bbdd_images, None, None)
    bbdd_descriptor = ORBDescriptor(bbdd_images,None,None)
    bbdd_results = bbdd_descriptor.compute_descriptors()
    print("Done. Time: "+str(time.time()-start))

    ## FINDING AND SORTING MATCHES
    threshold_distance = 250
    print("\nFinding and sorting matches...")
    start = time.time()
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = {}
    for img_key,img_res in query_results.items():
        # print("img_key:",img_key)
        matches[img_key] = []
        for paint_ind,paint_res in enumerate(img_res):
            matches[img_key].append([])
            # print("\tpaint_ind:",paint_ind)
            for bbdd_key,bbdd_img in bbdd_results.items():
                # print("\t\tbbdd_key:",bbdd_key,end=" ")
                if paint_res[1] is None or bbdd_img[0][1] is None:
                    matches[img_key][paint_ind].append(0)
                else:
                    m = matcher.match(paint_res[1], bbdd_img[0][1])
                    m = sorted(m, key=lambda x:x.distance)
                    m_keys = [item.distance for item in m]
                    m = m[:bisect.bisect_right(m_keys,threshold_distance)]
                    matches[img_key][paint_ind].append(len(m))
    print("Done. Time: "+str(time.time()-start))

    ## FINDING BEST MATCH FOR EACH PAINTING
    threshold_min_matches = 4
    max_best = 10
    print("\nFinding best match for each painting...")
    start = time.time()
    results = []
    for img_key,img_res in matches.items():
        results.append([])
        for paint_ind,paint_res in enumerate(img_res):
            m = matches[img_key][paint_ind]
            maxs = max(m)
            if maxs >= threshold_min_matches:
                best = [ind for ind,item in sorted(enumerate(m),reverse=True,key=lambda x:x[1])][:max_best]
            else:
                best = [-1]
            results[-1].append(best)
    print("Done. Time: "+str(time.time()-start))

    for ind,val in enumerate(results):
        print(ind,val)

    ## EVALUATING RESULTS
    print("\nEvaluating results...")
    start = time.time()
    evaluator = EvaluateDescriptors(results,qs_corresps_path)
    mapat1 = evaluator.compute_mapatk(limit=1)
    print("\tmap@1 = "+str(mapat1))
    print("Done. Time: "+str(time.time()-start))

    print("\nTotal time: "+str(time.time()-global_start))


if __name__ == "__main__":
    # main_qs2w3(False)

    main_qs1w4()
