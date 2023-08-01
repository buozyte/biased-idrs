import torch


def linear_bias_train(labels, orthogonal_vector, input_dim=2):
    """

    :param labels:
    :param orthogonal_vector:
    :param input_dim:
    :return:
    """

    bias = labels.clone()
    bias[bias == 0] = -1
    bias = bias.unsqueeze(dim=1).repeat_interleave(input_dim, dim=1)

    return bias * orthogonal_vector


def bias_linear_certify(x_index, oracles, base_classifier):
    """

    :return:
    """
    weight = None
    for name, param in base_classifier.named_parameters():
        if "weight" in name:
            weight = param.data
    w = (weight[1, 0] - weight[0, 0]) / (weight[0, 1] - weight[1, 1])
    orthogonal_vector = torch.tensor([-w, 1])

    return oracles[x_index] * orthogonal_vector
