import numpy as np
import torch

VARIANCE_FUNCTIONS = [None, "sigma_knn"]


def sigma_knn(sigma, rate, m, distances, x_index):
    """
    Input-dependent function to compute the variance for the sampling based on
    the k-nearest neighbours. Based on function proposed in: intriguing properties
    of input-dependent RS

    :param sigma: base value of variance
    :param rate: semi-elasticity constant for chosen sigma function
    :param m: normalization constant for data set
    :param distances: mean distances for every sample to its k nearest neighbours
    :param x_index: index of the current point
    :return: variance w.r.t. current input
    """

    return sigma * np.exp(rate * (distances[x_index] - m))

def sigma_knn_fcn(sigma, rate, m, mean_distances_fcn, x):
    """
    Input-dependent function to compute the variance for the sampling based on
    the k-nearest neighbours. Based on function proposed in: intriguing properties
    of input-dependent RS

    :param sigma: base value of variance
    :param rate: semi-elasticity constant for chosen sigma function
    :param m: normalization constant for data set
    :param mean_distances_fcn: function returning mean distances to the k nearest neighbours
    :param x: the current input
    :return: callable object that returns the variance for given x
    """

    return sigma * torch.exp(rate * (mean_distances_fcn(x) - m))
