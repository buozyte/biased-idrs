import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_blobs


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class ToyDatasetBlobsTrain(Dataset):
    """
    Clustered data samples.
    """

    def __init__(self, num):
        data, labels = make_blobs(n_samples=num, centers=2, random_state=42, cluster_std=1)

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


class ToyDatasetBlobsTest(Dataset):
    """
    Clustered data samples.
    """

    def __init__(self, num):
        data, labels = make_blobs(n_samples=num, centers=2, random_state=42, cluster_std=3)

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