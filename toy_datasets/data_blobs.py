import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_blobs
import os
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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


class ToyDatasetBlobsTrain(Dataset):
    """
    Clustered data samples.
    """

    def __init__(self, num):
        data, labels = make_blobs(n_samples=num, centers=4, random_state=42, cluster_std=1)

        shuffling = np.random.permutation(len(labels))
        data = data[shuffling]
        labels = labels[shuffling]

        data = torch.Tensor(data)
        labels = torch.Tensor(labels)  # .reshape(-1, 1)

        self.data = data
        self.labels = labels

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def __len__(self):
        return len(self.labels)
    
    def vis_itself(self):
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels)


class ToyDatasetBlobsTest(Dataset):
    """
    Clustered data samples.
    """

    def __init__(self, num):
        data, labels = make_blobs(n_samples=num, centers=4, random_state=42, cluster_std=3)

        shuffling = np.random.permutation(len(labels))
        data = data[shuffling]
        labels = labels[shuffling]

        data = torch.Tensor(data)
        labels = torch.Tensor(labels)  # .reshape(-1, 1)

        self.data = data
        self.labels = labels

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def __len__(self):
        return len(self.labels)
    
    def vis_itself(self):
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels)