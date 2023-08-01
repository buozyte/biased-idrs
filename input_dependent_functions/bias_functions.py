import numpy as np
import torch

BIAS_FUNCTIONS = [None, "mu_toy"]

def mu_toy(oracles, bias_weight, x_index, base_classifier):
    """
    Input-dependent function to compute a bias for the toy example using a linear classifier.
    Note that this bias function is not Lipschitz continuous.
    
    :param oracles: output oracle for each sample based on the k nearest neighbours
    :param bias_weight: "weight" of the bias
    :param x_index: index of the current point
    :param base_classifier: the base classifier
    :return: bias w.r.t. current input
    """
        
    weight = None
    for name, param in base_classifier.named_parameters():
        if "weight" in name:
            weight = param.data
    w = (weight[1, 0] - weight[0, 0]) / (weight[0, 1] - weight[1, 1])
    orthogonal_vector = torch.tensor([-w, 1])

    return bias_weight * orthogonal_vector * oracles[x_index]

def mu_toy_train(labels, orthogonal_vector, input_dim=2):
    """
    Bias function used for the training of the toy example.

    :param labels: labels of the data
    :param orthogonal_vector: suitable orthogonal vector
    :param input_dim: dimension of the input data space
    :return: biases (w.r.t. the given sample)
    """

    bias = labels.clone()
    bias[bias == 0] = -1
    bias = bias.unsqueeze(dim=1).repeat_interleave(input_dim, dim=1)

    return bias * orthogonal_vector
