# -- CLASS TO MAKE THE SEARCH -- #

# import the necessary packages
import random
import textdistance
import distance_metrics
import numpy as np
import pickle
import os
import time

class Searcher():
    """CLASS::SEARCHER:
        >- Class to search the top K most similar images given the database and query features."""
    def __init__(self,data_desc,query_desc):
        self.data = data_desc
        self.query = query_desc
        self.result = []

    def search(self,limit=3):
        """METHOD::SEARCH
            Searches the k number of features more similar from the query set."""
        # iterate through the query features
        print('--- SEARCHING MOST SIMILAR --- ')
        for qimg,qfeat in self.query.items():
            retrieve = []
            for ft in qfeat:
                distances = []
                # iterate through the db features
                for dimg,dfeat in self.data.items():
                    # compute distance
                    result = {'name':dimg,'dist':distance_metrics.chi2_distance(ft,dfeat)}
                    distances.append(result)
                # make a list with all the distances from one query
                less_dist = sorted(distances, key=lambda k: k['dist'])
                # get the first limit images from the db for that query image
                coincidences = [less_dist[k]['name'] for k in range(limit)]
                retrieve.append(coincidences)
            self.result.append(retrieve)
            print('Image ['+str(qimg)+'] Processed.')
            print('-------')
        print('--- DONE --- ')
        
    def clear_memory(self):
        """METHOD::CLEAR_MEMORY:
            >- Deletes the memory allocated that stores data to make it more efficient."""
        self.result = []

class SearcherText(Searcher):
    """CLASS::SearcherText:
        >- Class to search the top K most similar images given the database and query features."""
    def __init__(self,data_desc,query_desc):
        super(SearcherText,self).__init__(data_desc,query_desc)

    def search(self,limit=3):
        """METHOD::SEARCH
            Searches the k number of features more similar from the query set.
            Distances are hamming,jaccard or levenshtein"""
        # iterate through the query features
        print('--- SEARCHING MOST SIMILAR --- ')
        for qimg,qfeat in self.query.items():
            retrieve = []
            for ft in qfeat:
                distances = []
                # iterate through the db features
                for dimg,dfeat in self.data.items():
                    # compute distance
                    dist = textdistance.levenshtein.normalized_similarity(ft[0][0], dfeat[0][0])
                    result = {'name':dimg,'dist':1-dist}
                    distances.append(result)
                # make a list with all the distances from one query
                less_dist = sorted(distances, key=lambda k: k['dist'])
                # get the first limit images from the db for that query image
                coincidences = [less_dist[k]['name'] for k in range(limit)]
                retrieve.append(coincidences)
            self.result.append(retrieve)
            print('Image ['+str(qimg)+'] Processed.')
            print('-------')
        print('--- DONE --- ')

class SearcherCombined():
    """CLASS::SearcherText:
        >- Class to search the top K most similar images given the database and query features."""
    def __init__(self,data_desc1,query_desc1,data_desc2,query_desc2, data_text, query_text, use_text=False):
        self.data1 = data_desc1
        self.data2 = data_desc2
        self.query1 = query_desc1
        self.query2 = query_desc2
        self.query_text = query_text
        self.data_text = data_text
        self.use_text = use_text
        self.result = []

    def search(self,limit=3):
        """METHOD::SEARCH
            Searches the k number of features more similar from the query set.
            Distances are hamming,jaccard or levenshtein"""
        # iterate through the query features
        print('--- SEARCHING MOST SIMILAR --- ')
        for qimg,((_,q1),(_,q2)) in enumerate(zip(self.query1.items(),self.query2.items())):
            retrieve = []
            for ft1,ft2 in zip(q1,q2):
                distances = []
                # iterate through the db features
                for l,((_,d1),(_,d2)) in enumerate(zip(self.data1.items(),self.data2.items())):
                    # compute distance
                    for fd1,fd2 in zip(d1,d2):
                        result1 = distance_metrics.chi2_distance(ft1,fd1)
                        result2 = distance_metrics.chi2_distance(ft2,fd2)
                        extra_text_distance = 0
                        if self.use_text:
                            q_text = self.query_text[qimg][0][0][0]
                            b_text = self.data_text[l][0][0]
                            if not q_text.strip() or textdistance.levenshtein.normalized_similarity(q_text, b_text) < 0.85:
                                extra_text_distance = 1000000
                        result = {'name':l,'dist':result1+result2+extra_text_distance}
                        distances.append(result)
                # make a list with all the distances from one query
                less_dist = sorted(distances, key=lambda k: k['dist'])
                # get the first limit images from the db for that query image
                coincidences = [less_dist[k]['name'] for k in range(limit)]
                retrieve.append(coincidences)
            self.result.append(retrieve)
            print('Image ['+str(qimg)+'] Processed.')
            print('-------')
        print('--- DONE --- ')
    
    def clear_memory(self):
        """METHOD::CLEAR_MEMORY:
            >- Deletes the memory allocated that stores data to make it more efficient."""
        self.result = []