# -- CLASS TO MAKE THE SEARCH -- #

# import the necessary packages
from MAPatK import MAPatK
import distance_metrics
import numpy as np
import pickle
import os


class Searcher():
	"""CLASS::SEARCHER:
		>- Class to search the top K most similar images given the database and query features."""
	def __init__(self, data_path, query_path):
		with open(data_path,'rb') as data_file:
			self.data = pickle.load(data_file)
		with open(query_path,'rb') as query_file:
			self.query = pickle.load(query_file)
		self.results = []

	def search(self,limit=10):
		"""METHOD::SEARCH
			Searches the k number of features more similar from the query set."""
		# iterate through the query features
		print('--- SEARCHING MOST SIMILAR --- ')
		print('-------')
		for qimg,qfeat in self.query:
			distances = []
			# iterate through the db features
			for dimg,dfeat in self.data:
				# compute distance
				result = {'name':dimg,'dist':distance_metrics.chi2_distance(qfeat,dfeat)}
				distances.append(result)
			# make a list with all the distances from one query
			less_dist = sorted(distances, key=lambda k: k['dist'])
			# get the first 10 images from the db for that query image
			retrieve = [less_dist[k]['name'] for k in range(limit)]
			retrieve.insert(0,qimg)
			self.results.append(retrieve)
			print('Image ['+str(qimg)+'] Processed.')
			print('-------')

	def save_results(self,out_path,filename):
		"""METHOD::SAVE_METHODS:
			>- Save the results of the search engine."""
		with open(out_path+os.sep+filename,'wb') as file:
			pickle.dump(self.results,file)
		print('--- SEARCH RESULTS SAVED ---')

