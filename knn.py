import torch
from torch.utils.data import DataLoader


class KNNDistComp:
    """
    Compute the distance of one or multiple samples to its/their k nearest neighbours (in the set data set)
    """

    def __init__(self, main_data, num_workers, device):
        """
        :param main_data:
        :param num_workers:
        :param device:
        """

        self.main_data = main_data
        self.num_samples = len(main_data)
        self.main_dataloader = DataLoader(self.main_data,
                                          shuffle=False,
                                          batch_size=self.num_samples,
                                          num_workers=num_workers,
                                          pin_memory=False)
        self.device = device

    def _obtain_data(self):
        """

        :return: obtain the
        """

        data = None
        for (data_, _) in self.main_dataloader:
            data = data_.to(self.device)
        data.requires_grad = False
        return data

    def _obtain_data_with_labels(self):
        """

        :return: obtain the
        """

        data = None
        labels = None
        for (data_, labels_) in self.main_dataloader:
            data = data_.to(self.device)
            labels = labels_.to(self.device)
        data.requires_grad = False
        labels.requires_grad = False
        return data, labels

    def compute_1nn_oracle(self, data, norm=2):
        """

        :param data:
        :param norm:
        :return:
        """

        data = data.to(self.device)
        raw_data, raw_labels = self._obtain_data_with_labels()

        dists = torch.cdist(data.reshape((len(data), -1)),
                            raw_data.reshape((len(self.main_data), -1)), p=norm)  # .to(self.device)

        sorted_indices = dists.argsort(dim=1)
        return raw_labels[sorted_indices[:, 0]]

    def compute_knn_oracle(self, data, k=5, norm=2):
        """

        :param data:
        :param norm:
        :return:
        """

        data = data.to(self.device)
        raw_data, raw_labels = self._obtain_data_with_labels()

        dists = torch.cdist(data.reshape((len(data), -1)),
                            raw_data.reshape((len(self.main_data), -1)), p=norm)  # .to(self.device)

        sorted_indices = dists.argsort(dim=1)
        oracles, _ = torch.mode(raw_labels[sorted_indices[:, 0:k]], dim=1)
        return oracles

    def compute_dist(self, data, k, norm=2):
        """

        :param data: input for which the distances should be computed
        :param k: number of nearest neighbours to be considered
        :param norm: definition of the used lp norm
        :return: mean of the distances to the k nearest neighbours for each element in `data`
        """

        data = data.to(self.device)
        raw_data = self._obtain_data().to(self.device)

        dists = torch.cdist(data.reshape((len(data), -1)),
                            raw_data.reshape((len(self.main_data), -1)), p=norm)  # .to(self.device)

        sorted_dists, _ = dists.sort(dim=1)
        knn_means = sorted_dists[:, :k].mean(dim=1)
        return knn_means
