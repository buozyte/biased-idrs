import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import Dataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.random.seed(5)

# SEPARATOR = np.array([[-0.11588811,  1.11743863], [2.65907856,  0.18417988]])
SEPARATOR = np.array([[-0.1,  -0.1], [0.1,  0.1]])
ORTHOGONAL = [-1, 1]
M = (SEPARATOR[0, 1] - SEPARATOR[1, 1]) / (SEPARATOR[0, 0] - SEPARATOR[1, 0])
C = SEPARATOR[0, 1] - M * SEPARATOR[0, 0]


def generate_lin_separable_data(num, high, low, p):
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


def visualize_dataset_with_classifier(data, labels, ls, model, show=False, save=False, file_path="", add_file_name="",
                                      train=True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_1 = data[labels == 1]
    data_2 = data[labels == 0]

    if train:
        plt.figure(2)
    else:
        plt.figure(3)
    plt.scatter(data_1[:, 0], data_1[:, 1], label="class: 1", s=3, color='blue')
    plt.scatter(data_2[:, 0], data_2[:, 1], label="class: 0", s=3, color='red')

    plt.scatter(data_1[:, 0] + 0.3 * ORTHOGONAL[0], data_1[:, 1] + 0.3 * ORTHOGONAL[1],
                label="class: 1 (biased)", s=3, color='blue', marker='D', alpha=0.5)
    plt.scatter(data_2[:, 0] - 0.3 * ORTHOGONAL[0], data_2[:, 1] - 0.3 * ORTHOGONAL[1],
                label="class: 0 (biased)", s=3, color='red', marker='D', alpha=0.5)

    plt.plot(ls, M * ls + C, 'r-')

    xx, yy = np.meshgrid(ls, ls)
    inputs = np.c_[xx.ravel(), yy.ravel()]
    outputs = model(torch.from_numpy(inputs).to(torch.float32).to(device))
    _, pred = outputs.topk(1, 1, True, True)
    pred = pred.t()
    plt.contourf(xx, yy, pred.reshape(xx.shape), cmap=plt.cm.Spectral, alpha=0.1)

    plt.legend(fontsize=8)
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

    def visualize_with_classifier(self, model, save=True, file_path="", add_file_name="", show=False):
        ls = np.linspace(self.low-1, self.high+1, num=300)

        visualize_dataset_with_classifier(self.data, self.labels, ls, model, show, save, file_path, add_file_name, True)

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

    def visualize_with_classifier(self, model, show=False, save=True, file_path="", add_file_name=""):
        ls = np.linspace(self.low-1, self.high+1, num=300)

        visualize_dataset_with_classifier(self.data, self.labels, ls, model, show, save, file_path, add_file_name, False)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def __len__(self):
        return len(self.labels)
