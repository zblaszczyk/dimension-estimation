import numpy as np
import random
import sys

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import kneighbors_graph


def mean_sample_length(dataset, n_points, n_neighbors, n_resamplings):
	""" Samples `n_points` from `dataset` `n_resampling` times and calculates 
		the mean total edge length (i.e. the sum of lengths of all the edges) 
		of the resulting kNN graphs for k = `n_neighbours`. """
	length = 0
	for _ in range(n_resamplings):
		sample = random.sample(range(len(dataset)), n_points)
		sample_kNN = kneighbors_graph(dataset[sample], 5, mode='distance', include_self=False, n_jobs=-1)
		length += np.sum(sample_kNN)
	return length/n_resamplings


def estimator(dataset, routine, n_neighbors=5, n_runs=1, n_resamplings=10):
	""" Estimates the intrinsic dimension of `dataset` based on the length of its kNN graph, where k = `n_neighbors`. """
	routine_length = len(routine)
	estimates = []	
	X = np.ones((routine_length, 2))
	X[:, 0] = [np.log(x) for x in routine]
	for _ in range(n_runs):
		mean_lengths = np.zeros(routine_length)
		for index, n_points in enumerate(routine):
			mean_lengths[index] = np.log(mean_sample_length(dataset, n_points, n_neighbors, n_resamplings)) 
		model = LinearRegression().fit(X, mean_lengths)
		estimates.append(1/(1-model.coef_[0]))
	return int(round(sum(estimates)/n_runs))
