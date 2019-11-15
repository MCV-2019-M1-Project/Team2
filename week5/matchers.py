import cv2
import numpy as np
from glob import glob
import bisect
import os

class MatcherBF:
    """CLASS::Matcher:
        >- Class to search and match the top K most similar images given the database and query features.
        Possible Measures:
            >- cv2.NORM_L2. (DEFAULT)
            >- cv2.NORM_L2SQR.
            >- cv2.NORM_L1.
            >- cv2.NORM_HAMMING (useful for ORB).
            >- cv2.NORM_MINMAX."""
    def __init__(self, data_desc, query_desc, measure=cv2.NORM_HAMMING):
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
            self.result.append([])
            for paint_ind, paint_res in enumerate(img_res):
                matches = []
                for data_key,data_img in self.data.items():
                    if paint_res[1] is None or data_img[0][1] is None:
                        matches.append({'name':data_key,'num':0})
                    else:
                        m = self.matcher.match(paint_res[1], data_img[0][1])
                        m_sorted = sorted(m, key = lambda x:x.distance)
                        m_sorted = [item.distance for item in m_sorted]
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

    def draw_matches(self, img1, img2, ind_img, ind_paint, ind_db, name, num_matches=25):
        """FUNCTION TO DRAW MATCHES FOR THE SLIDES. 
            >- Need to first know the images that correspond between each other.
            >- pass the images"""
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        kp_img = self.query[ind_img][ind_paint][0]
        des_img = self.query[ind_img][ind_paint][1]
        kp_db = self.data[ind_db][0][0]
        key1 = []
        for point in kp_img:
            x = point[0]
            y = point[1]
            key1.append(cv2.KeyPoint(x, y, _size=3))
        key2 = []
        for point in kp_db:
            x = point[0]
            y = point[1]
            key2.append(cv2.KeyPoint(x,y,_size=3))
        des_db = self.data[ind_db][0][1]
        matches = self.matcher.match(des_img,des_db)
        matches = sorted(matches, key = lambda x:x.distance)
        img = cv2.drawMatches(img1,key1,img2,key2,matches[:num_matches],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(os.getcwd()+os.sep+name,img)

class MatcherFLANN:
    """CLASS::Matcher:
        >- Class to search and match the top K most similar images given the database and query features.
        Possible Measures:
            >- cv2.NORM_L2. (DEFAULT)
            >- cv2.NORM_L2SQR.
            >- cv2.NORM_L1.
            >- cv2.NORM_HAMMING (useful for ORB).
            >- cv2.NORM_MINMAX."""
    def __init__(self, data_desc, query_desc,flag=True):
        self.data = data_desc
        self.query = query_desc
        self.result = []
        """>- FLANN_INDEX_KDTREE = 1, while using SIFT OR SURF.
           >- FLANN_INDEX_LSH = 6, while using ORB."""
        if flag:
            FLANN_INDEX_LSH = 6
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
        else:
            FLANN_INDEX_KDTREE = 1
            index_params= dict(algorithm = FLANN_INDEX_KDTREE,
                             trees = 5)
        search_param = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params,search_param)
    
    def match(self, limit=10, min_matches=28, match_ratio=0.65):
        """METHOD::SEARCH
            Matches the k number of features more similar from the query set.
            Depending on the descriptor, threshold should be different."""
        print('--- MATCHING AND SEARCHING MOST SIMILAR --- ')
        for img_key, img_res in self.query.items():
            self.result.append([])
            for paint_ind, paint_res in enumerate(img_res):
                matches = []
                for data_key,data_img in self.data.items():
                    if paint_res[1] is None or data_img[0][1] is None:
                        matches.append({'name':data_key,'num':0})
                    else:
                        m = self.matcher.knnMatch(paint_res[1], data_img[0][1], k=2)
                        if len(m) > 0:
                            good_matches = []
                            try:
                                for n1,n2 in m:
                                    if n1.distance < match_ratio*n2.distance:
                                        good_matches.append(n1)
                                matches.append({'name':data_key,'num':len(good_matches)})
                            except ValueError as e:
                                matches.append({'name':data_key,'num':0})
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
        print("result:",self.result)

    def draw_matches(self, img1, img2, ind_img, ind_paint, ind_db, name, num_matches=25):
        """FUNCTION TO DRAW MATCHES FOR THE SLIDES. 
            >- Need to first know the images that correspond between each other.
            >- pass the images"""
        img1 = cv2.resize(cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY),(512,512))
        img2 = cv2.resize(cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY),(512,512))
        kp_img = self.query[ind_img][ind_paint][0]
        des_img = self.query[ind_img][ind_paint][1]
        kp_db = self.data[ind_db][0][0]
        key1 = []
        for point in kp_img:
            x = point[0]
            y = point[1]
            key1.append(cv2.KeyPoint(x, y, _size=3))
        key2 = []
        for point in kp_db:
            x = point[0]
            y = point[1]
            key2.append(cv2.KeyPoint(x,y,_size=3))
        des_db = self.data[ind_db][0][1]
        matches = self.matcher.match(des_img,des_db)
        matches = sorted(matches, key = lambda x:x.distance)
        img = cv2.drawMatches(img1,key1,img2,key2,matches[:num_matches],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(os.getcwd()+os.sep+name,img)
