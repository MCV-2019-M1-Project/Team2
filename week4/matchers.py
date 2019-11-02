import cv2
import numpy as np
from glob import glob
import bisect
import os

"""EQUIVALENT CLASS TO THE SEARCHER CLASS FOR OTHER DESCRIPTORS """
class MatcherBF:
    """CLASS::Matcher:
        >- Class to search and match the top K most similar images given the database and query features.
        Possible Measures:
            >- cv2.NORM_L2. (DEFAULT)
            >- cv2.NORM_L2SQR.
            >- cv2.NORM_L1.
            >- cv2.NORM_HAMMING (useful for ORB).
            >- cv2.NORM_MINMAX."""
    def __init__(self, data_desc, query_desc, measure=cv2.NORM_L2):
        self.data = data_desc
        self.query = query_desc
        self.result = []
        self.matcher = cv2.BFMatcher(measure,crossCheck=True)
    
    def match(self, limit=10, min_matches=4, threshold_distance=400):
        """METHOD::SEARCH
            Matches the k number of features more similar from the query set.
            Depending on the descriptor, threshold should be different."""
        print('--- MATCHING AND SEARCHING MOST SIMILAR --- ')
        for img_key, img_res in self.query.items():
            print("img_key:",img_key)
            self.result.append([])
            for paint_ind, paint_res in enumerate(img_res):
                print("\tpaint_ind:",paint_ind)
                try:
                    print("\tpaint_res[1].shape:",paint_res[1].shape)
                except:
                    print("\tpaint_res[1] has no shape")
                matches = []
                for data_key,data_img in self.data.items():
                    if paint_res[1] is None or data_img[0][1] is None:
                        matches.append({'name':data_key,'num':0})
                    else:
                        m = self.matcher.match(paint_res[1], data_img[0][1])
                        m_sorted = sorted([item.distance for item in m])
                        num_matches = len(m[:bisect.bisect_right(m_sorted,threshold_distance)])
                        matches.append({'name':data_key,'num':num_matches})
                candidates = sorted(matches, reverse=True, key=lambda k: k['num'])
                if candidates[0]['num'] >= min_matches:
                    coincidences = []
                    for k in range(limit):
                        try:
                            coincidences.append(candidates[k]['name'])
                        except:
                            pass
                else:
                    """RETURN -1 AS IT IS NOT ON THE DATABASE."""
                    coincidences = [-1]*limit
                self.result[-1].append(coincidences)
            print('Image ['+str(img_key)+'] Processed.')
            print('-------')
        print('--- DONE --- ')
        print("result:",self.result)

    def draw_matches(self, img1, img2, ind_img, ind_paint, ind_db, num_matches=10):
        """FUNCTION TO DRAW MATCHES FOR THE SLIDES. 
            >- Need to first know the images that correspond between each other.
            >- pass the images"""
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        kp_img = self.query[ind_img][ind_paint][0]
        des_img = self.query[ind_img][ind_paint][1]
        kp_db = self.data[ind_db][0][0]
        des_db = self.data[ind_db][0][1]
        matches = self.matcher.match(des_img,des_db)
        matches = sorted(matches, key = lambda x:x.distance)
        img = cv2.drawMatches(img1,kp_img,img2,kp_db,matches[:num_matches], flags=2)
        cv2.imwrite(os.getcwd()+os.sep+'M.png',img)


class MatcherFLANN:
    """CLASS::Matcher:
        >- Class to search and match the top K most similar images given the database and query features.
        Possible Measures:
            >- cv2.NORM_L2. (DEFAULT)
            >- cv2.NORM_L2SQR.
            >- cv2.NORM_L1.
            >- cv2.NORM_HAMMING (useful for ORB).
            >- cv2.NORM_MINMAX."""
    def __init__(self, data_desc, query_desc, measure=cv2.NORM_L2):
        self.data = data_desc
        self.query = query_desc
        self.result = []
        """>- FLANN_INDEX_KDTREE = 1, while using SIFT OR SURF.
           >- FLANN_INDEX_LSH = 6, while using ORB."""
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                            table_number = 6, # 12
                            key_size = 12,     # 20
                            multi_probe_level = 1,
                            trees = 5) #2
        search_param = dict(checks=50)
        # FLANN_INDEX_KDTREE = 1
        # index_params= dict(algorithm = FLANN_INDEX_KDTREE,
        #                     trees = 5)
        # search_param = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params,search_param)
    
    def match(self, limit=10, min_matches=4, match_ratio=0.75):
        """METHOD::SEARCH
            Matches the k number of features more similar from the query set.
            Depending on the descriptor, threshold should be different."""
        print('--- MATCHING AND SEARCHING MOST SIMILAR --- ')
        for img_key, img_res in self.query.items():
            print("img_key:",img_key)
            self.result.append([])
            for paint_ind, paint_res in enumerate(img_res):
                print("\tpaint_ind:",paint_ind)
                try:
                    print("\tpaint_res[1].shape:",paint_res[1].shape)
                except:
                    print("\tpaint_res[1] has no shape")
                matches = []
                for data_key,data_img in self.data.items():
                    if paint_res[1] is None or data_img[0][1] is None:
                        matches.append({'name':data_key,'num':0})
                    else:
                        m = self.matcher.knnMatch(paint_res[1], data_img[0][1], k=2)
                        if len(m) > 0:
                            good_matches = []
                            for n1,n2 in m:
                                if n1.distance < match_ratio*n2.distance:
                                    good_matches.append([n1])
                            matches.append({'name':data_key,'num':len(good_matches)})
                        else:
                            matches.append({'name':data_key,'num':0})
                candidates = sorted(matches, reverse=True, key=lambda k: k['num'])
                if candidates[0]['num'] >= min_matches:
                    coincidences = [candidates[k]['name'] for k in range(limit)]
                else:
                    """RETURN -1 AS IT IS NOT ON THE DATABASE."""
                    coincidences = [-1]*limit
                self.result[-1].append(coincidences)
            print('Image ['+str(img_key)+'] Processed.')
            print('-------')
        print('--- DONE --- ')

    def draw_matches(self, img1, img2, ind_img, ind_paint, ind_db, num_matches=10):
        """FUNCTION TO DRAW MATCHES FOR THE SLIDES. 
            >- Need to first know the images that correspond between each other.
            >- pass the images"""
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        kp_img = self.query[ind_img][ind_paint][0]
        des_img = self.query[ind_img][ind_paint][1]
        kp_db = self.data[ind_db][0][0]
        des_db = self.data[ind_db][0][1]
        matches = self.matcher.match(des_img,des_db)
        matches = sorted(matches, key = lambda x:x.distance)
        img = cv2.drawMatches(img1,kp_img,img2,kp_db,matches[:num_matches], flags=2)
        cv2.imwrite(os.getcwd()+os.sep+'M.png',img)


# class MatcherRatio(Matcher):
#     def __init__(self, data_desc, query_desc, type_match='bf', measure=cv2.NORM_L2):
#         super().__init__(data_desc, query_desc, type_match=type_match, measure=measure)
    
#     def match(self, limit=10, min_matches=4, match_ratio=0.75):
#         print('--- MATCHING AND SEARCHING MOST SIMILAR --- ')
#         for img_key, img_res in self.query.items():
#             self.result.append([])
#             for paint_ind, paint_res in enumerate(img_res):
#                 matches = []
#                 for data_key,data_img in self.data.items():
#                     if paint_res[1] is None or data_img[0][1] is None:
#                         matches.append({'name':data_key,'num':0})
#                     else:
#                         m = self.matcher.match(paint_res[1], data_img[0][1])
#                         good_matches = []
#                         for n1,n2 in m:
#                             if n1.distance < match_ratio*n2.distance:
#                                 good_matches.append([n1])
#                         matches.append({'name':data_key,'num':len(good_matches)})
#                 candidates = sorted(matches, reverse=True, key=lambda k: k['num'])
#                 if candidates[0]['num'] >= min_matches:
#                     coincidences = [candidates[k]['name'] for k in range(limit)]
#                 else:
#                     """RETURN -1 AS IT IS NOT ON THE DATABASE."""
#                     coincidences = [-1]*limit
#                 self.result[-1].append(coincidences)
#             print('Image ['+str(img_key)+'] Processed.')
#             print('-------')
#         print('--- DONE --- ')
