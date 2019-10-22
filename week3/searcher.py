# -- CLASS TO MAKE THE SEARCH -- #

# import the necessary packages
import distance_metrics
import numpy as np
import pickle
import os
from multiprocessing import Process, Manager, Value, Lock
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


class SearcherMultiprocess():
    """CLASS::SEARCHER:
        >- Class to search the top K most similar images given the database and query features."""
    def __init__(self,data_desc,query_desc,num_cores):
        self.data = data_desc
        self.query = query_desc
        self.num_cores = num_cores
        self.result = Manager().list()
        self.lock = Lock()

    def search(self,limit=3):
        """METHOD::SEARCH
            Searches the k number of features more similar from the query set."""
        # iterate through the query features
        print('--- SEARCHING MOST SIMILAR --- ')
        query_items = list(self.query.items())
        id_item = 0
        while id_item < len(query_items):
            plist = []
            for i in range(self.num_cores):
                if id_item < len(query_items):
                    p = Process(target=self.search_process,args=(query_items[id_item],id_item,limit,self.result,self.lock))
                    p.daemon = True
                    plist.append(p)
                    id_item += 1
            for p in plist:
                p.start()
            for p in plist:
                p.join()
        print('--- DONE --- ')

    def search_process(self,q,id_item,limit,result,lock):
        qimg,qfeat = q
        retrieve = []
        for ft in qfeat:
            distances = []
            # iterate through the db features
            for dimg,dfeat in self.data.items():
                # compute distance
                res = {'name':dimg,'dist':distance_metrics.chi2_distance(ft,dfeat)}
                distances.append(res)
            # make a list with all the distances from one query
            less_dist = sorted(distances, key=lambda k: k['dist'])
            # get the first limit images from the db for that query image
            coincidences = [less_dist[k]['name'] for k in range(limit)]
            retrieve.append(coincidences)
        lock.acquire()
        result.append(retrieve)
        lock.release()
        print('Image ['+str(qimg)+'] Processed.')
        print('-------')
        
    def clear_memory(self):
        """METHOD::CLEAR_MEMORY:
            >- Deletes the memory allocated that stores data to make it more efficient."""
        self.result = []

