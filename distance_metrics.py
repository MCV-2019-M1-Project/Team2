# -- DISTANCE_METRICS -- #

# -- IMPORTS -- #
import numpy as np
import math as m

# -- ARRAYS MUST BEOF SIZE (X,1) -- #

# -- DISTANCE METRICS -- #
def euclidean_distance(x,y):
    """FUNCTION::EUCLIDEAN_DISTANCE
        >- Returns: The euclidean distance between the two arrays."""
    return np.sqrt(np.sum(np.power(np.subtract(x,y),2),axis=-1))

def manhattan_distance(x,y):
	"""FUNCTION::MANHATTAN_DISTANCE:
		>- Returns: The manhattan distance (also known as L1 norm) between the two arrays."""
	return np.sum(np.abs(np.subtract(x,y)),axis=-1)
	
def cosine_distance(x,y):
	"""FUNCTION::COSINE_DISTANCE:
        >- Returns: The cosine distance between two arrays."""
	return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

def chi_squared_distance(x,y):
    """FUNCTION::CHI_SQUARED_DISTANCE:
        >- Returns: The chi squared distance between two arrays.
        Works well with histograms."""
    return np.sum((np.power(np.subtract(x,y),2)/np.add(x,y)),axis=-1)

def histogram_intersection(x,y):
    """FUNCTION::HISTOGRAM_INTERSECTION:
        >- Returns: The histogram intersection between two arrays.
        Works well with histograms."""
    return np.sum(np.minimum(x,y),axis=-1)

def hellinger_kernel(x,y):
    """FUNCTION::HELLINGER_KERNEL:
        >- Returns: The hellinger kernel between two arrays."""
    return np.sum(np.sqrt(np.multiply(x,y)))
