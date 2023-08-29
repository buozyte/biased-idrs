import torch
import numpy as np

BIAS_FUNCTIONS = [None, "mu_toy", "mu_knn_based"]


def mu_toy(oracles, bias_weight, x_index, base_classifier, device):
    """
    Input-dependent function to compute a bias for the toy example using a linear classifier.
    Note that this bias function is not Lipschitz continuous.
    
    :param oracles: output oracle for each sample based on the k nearest neighbours
    :param bias_weight: "weight" of the bias
    :param x_index: index of the current point
    :param base_classifier: the base classifier
    :param device: pytorch device handling
    :return: bias w.r.t. current input
    """
        
    weight = None
    for name, param in base_classifier.named_parameters():
        if "weight" in name:
            weight = param.data
    w = (weight[1, 0] - weight[0, 0]) / (weight[0, 1] - weight[1, 1])
    orthogonal_vector = torch.tensor([-w, 1]).to(device)

    return bias_weight * orthogonal_vector * oracles[x_index]


def mu_toy_train(labels, orthogonal_vector, device, input_dim=2):
    """
    Bias function used for the training of the toy example.

    :param labels: labels of the data
    :param orthogonal_vector: suitable orthogonal vector
    :param device: pytorch device handling
    :param input_dim: dimension of the input data space
    :return: biases (w.r.t. the given sample)
    """

    bias = labels.clone()
    bias[bias == 0] = -1
    bias = bias.unsqueeze(dim=1).repeat_interleave(input_dim, dim=1)
    orthogonal_vector = orthogonal_vector.to(device)

    return bias * orthogonal_vector


def mu_nearest_neighbour(x, x_index, nearest_neighbours, distances, device, bias_weight=1):
    """
    Input-dependent function to compute a bias based on the direction from the current sample to it's nearest neighbour.

    :param x: current sample
    :param x_index: index of the current point
    :param nearest_neighbours: the k-nearest neighbours of x
    :param distances: the according distance to the k-nearest neighbours
    :param device: pytorch device handling
    :param bias_weight: "weight" of the bias
    :return: bias w.r.t. current input
    """

    x_distances = distances[x_index]
    dist_weight = np.min([x_distances[0], np.sqrt(x_distances[0] * (x_distances[1] - x_distances[0]))])

    return bias_weight * dist_weight * (nearest_neighbours[x_index][0].to(device) - x)
