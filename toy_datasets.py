import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import Dataset

# potential new toy dataset:
# sklearn.datasets.make_blobs(n_samples=num, n_features=2, centers=4, random_state=42)
# -> find suitable model?

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# SEPARATOR = np.array([[-0.11588811,  1.11743863], [2.65907856,  0.18417988]])
SEPARATOR = np.array([[-0.1,  -0.1], [0.1,  0.1]])
ORTHOGONAL = [-1, 1]
M = (SEPARATOR[0, 1] - SEPARATOR[1, 1]) / (SEPARATOR[0, 0] - SEPARATOR[1, 0])
C = SEPARATOR[0, 1] - M * SEPARATOR[0, 0]


def generate_lin_separable_data(num, high, low, p):
    np.random.seed(num)
    # generate random data in R^2
    data = np.random.uniform(low=low, high=high, size=(num, 2))

    # assign labels to data according to separation plane with uncertainty p
    eval_linear = M * data[:, 0] + C
    data_1 = data[data[:, 1] > eval_linear, :]
    labels_1 = np.random.binomial(1, p, len(data_1))
    # labels_1 = np.ones(len(data_1))
    data_2 = data[data[:, 1] <= eval_linear, :]
    labels_2 = np.random.binomial(1, 1 - p, len(data_2))
    # labels_2 = np.zeros(len(data_2))

    # put data and labels together
    data = np.concatenate((data_1, data_2), axis=0)
    labels = np.concatenate((labels_1, labels_2), axis=0)

    return data, labels


def visualize_dataset(data, labels, ls, show=False, train=True):
    data_1 = data[labels == 1]
    data_2 = data[labels == 0]

    if train:
        plt.figure(0)
    else:
        plt.figure(1)
    plt.scatter(data_1[:, 0], data_1[:, 1], label="class: 1", s=3)
    plt.scatter(data_2[:, 0], data_2[:, 1], label="class: 0", s=3)
    plt.plot(ls, M * ls + C, 'r-')

    plt.legend(fontsize=14)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)

    plt.xlim([min(ls)-0.2, max(ls)+0.2])
    plt.ylim([min(ls)-0.2, max(ls)+0.2])
    if train:
        plt.title("Train dataset", fontsize=16)
    else:
        plt.title("Test dataset", fontsize=16)
    if show:
        plt.show()


def visualize_dataset_with_classifier_oracle_based(data, labels, ls, model, bias_weight=0, oracles=None, show=False,
                                                   save=False, file_path="", add_file_name="", train=True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_1 = data[labels == 1]
    data_2 = data[labels == 0]

    if train:
        plt.figure(2)
    else:
        plt.figure(3)
    plt.scatter(data_1[:, 0], data_1[:, 1], label="class: 1", s=3, color='blue')
    plt.scatter(data_2[:, 0], data_2[:, 1], label="class: 0", s=3, color='red')

    if oracles is None and bias_weight > 0:
        plt.scatter(data_1[:, 0] + bias_weight * ORTHOGONAL[0], data_1[:, 1] + bias_weight * ORTHOGONAL[1],
                    label="class: 1 (biased)", s=3, color='blue', marker='D', alpha=0.5)
        plt.scatter(data_2[:, 0] - bias_weight * ORTHOGONAL[0], data_2[:, 1] - bias_weight * ORTHOGONAL[1],
                    label="class: 0 (biased)", s=3, color='red', marker='D', alpha=0.5)

        for i in range(0, len(data_1)):
            plt.plot([data_1[i, 0], data_1[i, 0] + bias_weight * ORTHOGONAL[0]],
                     [data_1[i, 1], data_1[i, 1] + bias_weight * ORTHOGONAL[1]], 'b-', alpha=0.2)

        for i in range(0, len(data_2)):
            plt.plot([data_2[i, 0], data_2[i, 0] - bias_weight * ORTHOGONAL[0]],
                     [data_2[i, 1], data_2[i, 1] - bias_weight * ORTHOGONAL[1]], 'r-', alpha=0.2)
    elif bias_weight > 0:
        weight = None
        for name, param in model.named_parameters():
            if "weight" in name:
                weight = param.data
        w = (weight[1, 0] - weight[0, 0]) / (weight[0, 1] - weight[1, 1])
        w = w.cpu()
        for i in range(0, len(data)):
            oracle = oracles[i]
            if oracle > 0:
                plt.plot([data[i, 0], data[i, 0] + bias_weight * oracle * (-w)],
                         [data[i, 1], data[i, 1] + bias_weight * oracle * 1], 'b-', alpha=0.2)
            else:
                plt.plot([data[i, 0], data[i, 0] + bias_weight * oracle * (-w)],
                         [data[i, 1], data[i, 1] + bias_weight * oracle * 1], 'r-', alpha=0.2)

    plt.plot(ls, M * ls + C, 'k-', alpha=0.1)

    xx, yy = np.meshgrid(ls, ls)
    inputs = np.c_[xx.ravel(), yy.ravel()]
    outputs = model(torch.from_numpy(inputs).to(torch.float32).to(device))
    _, pred = outputs.topk(1, 1, True, True)
    pred = pred.t().cpu()
    plt.contourf(xx, yy, pred.reshape(xx.shape), cmap=plt.cm.Spectral, alpha=0.1)

    plt.legend(fontsize=9)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)

    # plt.xlim([min(ls)-0.2, max(ls)+0.2])
    # plt.ylim([min(ls)-0.2, max(ls)+0.2])
    if train:
        plt.title("Train dataset", fontsize=16)
    else:
        plt.title("Test dataset", fontsize=16)
    if show:
        plt.show()
    if save:
        if train:
            plt.savefig(f"{file_path}/visual_decision_boundary_train{add_file_name}.pdf")
        else:
            plt.savefig(f"{file_path}/visual_decision_boundary_test{add_file_name}.pdf")


def visualize_dataset_with_classifier_knn_based(data, labels, ls, model, bias_weight=1, knns=None, distances=None,
                                                show=False, save=False, file_path="", add_file_name="", train=True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_1 = data[labels == 1]
    data_2 = data[labels == 0]

    if train:
        plt.figure(2)
    else:
        plt.figure(3)
    plt.scatter(data_1[:, 0], data_1[:, 1], label="class: 1", s=3, color='blue')
    plt.scatter(data_2[:, 0], data_2[:, 1], label="class: 0", s=3, color='red')

    if knns is not None and distances is not None:
        for i in range(0, len(data)):
            x_distances = distances[i]
            dist_weight = np.min([x_distances[0], np.sqrt(x_distances[0] * (x_distances[1] - x_distances[0]))])

            new_data = data[i, :] + bias_weight * dist_weight * (knns[i][0] - data[i, :])

            plt.plot([data[i, 0], new_data[0]],
                     [data[i, 1], new_data[1]], 'g-')

    plt.plot(ls, M * ls + C, 'k-', alpha=0.1)

    xx, yy = np.meshgrid(ls, ls)
    inputs = np.c_[xx.ravel(), yy.ravel()]
    outputs = model(torch.from_numpy(inputs).to(torch.float32).to(device))
    _, pred = outputs.topk(1, 1, True, True)
    pred = pred.t().cpu()
    plt.contourf(xx, yy, pred.reshape(xx.shape), cmap=plt.cm.Spectral, alpha=0.1)

    plt.legend(fontsize=9)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)

    # plt.xlim([min(ls)-0.2, max(ls)+0.2])
    # plt.ylim([min(ls)-0.2, max(ls)+0.2])
    if train:
        plt.title("Train dataset", fontsize=16)
    else:
        plt.title("Test dataset", fontsize=16)
    if show:
        plt.show()
    if save:
        if train:
            plt.savefig(f"{file_path}/visual_decision_boundary_train{add_file_name}.pdf")
        else:
            plt.savefig(f"{file_path}/visual_decision_boundary_test{add_file_name}.pdf")


class ToyDatasetLinearSeparationTrain(Dataset):
    """
    Linearly separated data.
    """

    def __init__(self, num, high, low, p=0.95):
        self.high = high
        self.low = low
        data, labels = self._make_dataset(num, p)

        shuffling = np.random.permutation(len(labels))
        data = data[shuffling]
        labels = labels[shuffling]

        data = torch.Tensor(data)
        labels = torch.Tensor(labels)  # .reshape(-1, 1)

        self.data = data
        self.labels = labels

    def _make_dataset(self, num, p):
        return generate_lin_separable_data(num, self.high, self.low, p)

    def visualize_itself(self, show=True):
        ls = np.linspace(self.low-1, self.high+1, num=300)

        visualize_dataset(self.data, self.labels, ls, show, True)

    def visualize_with_classifier_oracle_based(self, model, bias_weight=0, save=True, file_path="", add_file_name="",
                                               show=False):
        ls = np.linspace(self.low-1, self.high+1, num=300)

        visualize_dataset_with_classifier_oracle_based(self.data, self.labels, ls, model, bias_weight, None, show, save,
                                                       file_path, add_file_name, True)

    def visualize_with_classifier_knn_based(self, model, bias_weight=1, save=True, file_path="", add_file_name="",
                                            show=False):
        ls = np.linspace(self.low-1, self.high+1, num=300)

        visualize_dataset_with_classifier_knn_based(self.data, self.labels, ls, model, bias_weight, None, None, show,
                                                    save, file_path, add_file_name, True)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def __len__(self):
        return len(self.labels)


class ToyDatasetLinearSeparationTest(Dataset):
    """
    Linearly separated data.
    """

    def __init__(self, num, high, low, p=0.95):
        self.high = high
        self.low = low
        data, labels = self._make_dataset(num, p)

        shuffling = np.random.permutation(len(labels))
        data = data[shuffling]
        labels = labels[shuffling]

        data = torch.Tensor(data)
        labels = torch.Tensor(labels)

        self.data = data
        self.labels = labels

    def _make_dataset(self, num, p):
        return generate_lin_separable_data(num, self.high, self.low, p)

    def visualize_itself(self, show=True):
        ls = np.linspace(self.low-1, self.high+1, num=300)

        visualize_dataset(self.data, self.labels, ls, show, False)

    def visualize_with_classifier_oracle_based(self, model, bias_weight=0, oracles=None, show=False, save=True,
                                               file_path="", add_file_name=""):
        ls = np.linspace(self.low-1, self.high+1, num=300)

        visualize_dataset_with_classifier_oracle_based(self.data, self.labels, ls, model, bias_weight, oracles, show,
                                                       save, file_path, add_file_name, False)

    def visualize_with_classifier_knn_based(self, model, bias_weight=1, knns=None, distances=None, show=False,
                                            save=True, file_path="", add_file_name=""):
        ls = np.linspace(self.low-1, self.high+1, num=300)

        visualize_dataset_with_classifier_knn_based(self.data, self.labels, ls, model, bias_weight, knns, distances,
                                                    show, save, file_path, add_file_name, False)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def __len__(self):
        return len(self.labels)


# --------------------------------------------------
# second toy dataset (used in non-biased idrs paper)
# --------------------------------------------------


def generate_first_class(num, df, scale, angle):
    length = np.sqrt(np.random.chisquare(df=df, size=num) * scale)
    angle = np.random.uniform(low=-angle, high=angle, size=num)
    data_x = (np.cos(angle) * length).reshape((-1, 1))
    data_y = (np.sin(angle) * length).reshape((-1, 1))
    data = np.concatenate((data_x, data_y), axis=1)
    labels = np.zeros(num, dtype=int)
    return data, labels


def generate_second_class(num, df, scale, angle):
    length = np.sqrt(np.random.chisquare(df=df, size=num) * scale)
    angle = np.random.uniform(low=angle, high=2*np.pi-angle, size=num)
    data_x = (np.cos(angle) * length).reshape((-1, 1))
    data_y = (np.sin(angle) * length).reshape((-1, 1))
    data = np.concatenate((data_x, data_y), axis=1)
    labels = np.zeros(num, dtype=int) + 1
    return data, labels


def generate_both_classes(num, df, scale, angle):
    np.random.seed(num)
    np.random.seed(num)
    data_1, labels_1 = generate_first_class(num, df, scale, angle)
    data_2, labels_2 = generate_second_class(num, df, scale, angle)
    data = np.concatenate((data_1, data_2), axis=0)
    labels = np.concatenate((labels_1, labels_2), axis=0)
    return data, labels


def visualize_cone_dataset(data, labels, show=False, train=True):
    data_1 = data[labels == 0]
    data_2 = data[labels == 1]
    plt.scatter(data_1[:, 0], data_1[:, 1], label="first class", s=3)
    plt.scatter(data_2[:, 0], data_2[:, 1], label="second class", s=3)
    plt.legend(fontsize=14)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    if train:
        plt.title("Train dataset", fontsize=16)
    else:
        plt.title("Test dataset", fontsize=16)
    if show:
        plt.show()


class ToyDatasetConeShapedTrain(Dataset):
    """
    Cone-shaped dataset.
    """
    def __init__(self, num, df, scale, angle):
        data, labels = self._make_dataset(num, df, scale, angle)
        shuffling = np.random.permutation(len(labels))
        data = data[shuffling]
        labels = labels[shuffling]
        data = torch.Tensor(data)
        labels = torch.Tensor(labels)
        self.data = data
        self.labels = labels

    @staticmethod
    def _make_dataset(num, df, scale, angle):
        return generate_both_classes(num, df, scale, angle)

    def visualize_itself(self, show=False):
        visualize_cone_dataset(self.data, self.labels, show, True)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def __len__(self):
        return len(self.labels)


class ToyDatasetConeShapedTest(Dataset):
    """
    Cone-shaped dataset.
    """
    def __init__(self, num, df, scale, angle):
        data, labels = self._make_dataset(num, df, scale, angle)
        shuffling = np.random.permutation(len(labels))
        data = data[shuffling]
        labels = labels[shuffling]
        data = torch.Tensor(data)
        labels = torch.Tensor(labels)
        self.data = data
        self.labels = labels

    @staticmethod
    def _make_dataset(num, df, scale, angle):
        return generate_both_classes(num, df, scale, angle)

    def visualize_itself(self, show=False):
        visualize_cone_dataset(self.data, self.labels, show, False)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def __len__(self):
        return len(self.labels)
