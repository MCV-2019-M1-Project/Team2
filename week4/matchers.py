import cv2
import numpy as np
from glob import glob
import os

"""EQUIVALENT CLASS TO THE SEARCHER CLASS FOR OTHER DESCRIPTORS """
class Matcher:
    """CLASS::Matcher:
        >- Class to search and match the top K most similar images given the database and query features.
        Possible Measures:
            >- cv2.NORM_L2. (DEFAULT)
            >- cv2.NORM_L2SQR.
            >- cv2.NORM_L1.
            >- cv2.NORM_HAMMING (useful for ORB).
            >- cv2.NORM_MINMAX."""
    def __init__(self, data_desc, query_desc, type_match='bf', measure=cv2.NORM_L2):
        self.data = data_desc
        self.query = query_desc
        self.result = []
        if type_match is 'bf':
            self.matcher = cv2.BFMatcher(measure,crossCheck=True)
        elif type_match is 'flann':
            """>- FLANN_INDEX_KDTREE = 1, while using SIFT OR SURF.
               >- FLANN_INDEX_LSH = 6, while using ORB."""
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                                table_number = 6, # 12
                                key_size = 12,     # 20
                                multi_probe_level = 1,
                                trees = 5) #2
            search_param = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params,search_param)
        else:
            raise NotImplementedError
    
    def match(self, limit=10, min_matches=4, threshold=250):
        """METHOD::SEARCH
            Matches the k number of features more similar from the query set.
            Depending on the descriptor, threshold should be different."""
        print('--- MATCHING AND SEARCHING MOST SIMILAR --- ')
        for img_key, img_res in self.query.items():
            self.results.append([])
            for paint_ind, paint_res in enumerate(img_res):
                matches = []
                for data_key,data_img in self.data.items():
                    if paint_res is None or bbdd_img[0] is None:
                        matches.append({'name':data_key,'num':0})
                    else:
                        m = self.matcher.match(paint_res, bbdd_img[0])
                        num_matches = [(0 if item.distance <= threshold else 1) for item in m].count(0)
                        matches.append({'name':data_key,'num':num_matches})
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
                
        