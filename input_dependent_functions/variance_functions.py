import numpy as np

VARIANCE_FUNCTIONS = [None, "sigma_knn"]

def sigma_knn(sigma, rate, m, distances, x_index):
        """
        Input-dependent function to compute the variance for the sampling based on the k-nearest neighbours..
        Based on function proposed in: intriguing properties of input-dependent RS

        :param sigma: base value of variance
        :param rate: semi-elasticity constant for chosen sigma function
        :param m: normalization constant for data set
        :param distances: mean distances for every sample to its k nearest neighbours
        :param x_index: index of the current point
        :return: variance w.r.t. current input
        """

        return sigma * np.exp(rate * (distances[x_index] - m))