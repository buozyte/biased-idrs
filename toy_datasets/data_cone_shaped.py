import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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