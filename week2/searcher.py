# -- CLASS TO MAKE THE SEARCH -- #

# import the necessary packages
import distance_metrics
import numpy as np
import pickle
import os


class Searcher():
	"""CLASS::SEARCHER:
		>- Class to search the top K most similar images given the database and query features."""
	def __init__(self, data_desc, query_desc):
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
					result = {'name':dimg,'dist':distance_metrics.chi2_distance(qfeat,dfeat)}
					distances.append(result)
				# make a list with all the distances from one query
				less_dist = sorted(distances, key=lambda k: k['dist'])
				# get the first limit images from the db for that query image
				coincidences = [less_dist[k]['name'] for k in range(limit)]
				retrieve.append(coincidences)
			self.result.append(retrieve)
		print('--- DONE --- ')
		
	def clear_memory(self):
		"""METHOD::CLEAR_MEMORY:
			>- Deletes the memory allocated that stores data to make it more efficient."""
		self.result = []

