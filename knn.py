import torch
from torch.utils.data import DataLoader


class KNNDistComp:
    """
    Compute the distance of one or multiple samples to its/their k nearest neighbours (in the set data set)
    """

    def __init__(self, main_data, num_workers, device):
        """
        Init KNN computer
        
        :param main_data: base data for the computation of the k nearest neighbours
        :param num_workers: number of workers to be used
        :param device: device for device handling
        """

        self.main_data = main_data
        self.num_samples = len(main_data)
        self.main_dataloader = DataLoader(self.main_data,
                                          shuffle=False,
                                          batch_size=self.num_samples,
                                          #num_workers=num_workers,
                                          pin_memory=False)
        self.device = device

    def _obtain_data(self):
        """
        Set parameters for data objects and push them to correct device.

        :return: correct data
        """

        data = None
        for (data_, _) in self.main_dataloader:
            data = data_.to(self.device)
        data.requires_grad = False
        return data

    def _obtain_data_with_labels(self):
        """
        Set parameters for data and labels objects and push them to correct device.

        :return: correct data and labels
        """

        data = None
        labels = None
        for (data_, labels_) in self.main_dataloader:
            data = data_.to(self.device)
            labels = labels_.to(self.device)
        data.requires_grad = False
        labels.requires_grad = False
        return data, labels

    def compute_knn_oracle(self, data, k, norm=2):
        """
        Compute an oracle for the label of each data point based on the labels of the k nearest neighbours (w.r.t. the
        main data).

        :param data: input for which the distances should be computed
        :param k: number of nearest neighbours to be considered
        :param norm: definition of the used lp norm
        :return: oracle for labels
        """

        data = data.to(self.device)
        raw_data, raw_labels = self._obtain_data_with_labels()

        dists = torch.cdist(data.reshape((len(data), -1)),
                            raw_data.reshape((len(self.main_data), -1)), p=norm)  # .to(self.device)

        _, ids = dists.topk(k, dim=1, largest=False)

        return torch.stack([raw_labels[id] for id in ids])

    def compute_1nn_oracle(self, data, norm=2):
        """
        Compute an oracle for the label of each data point based on the label of the nearest neighbour (w.r.t. the main
        data).

        :param data: input for which the distances should be computed
        :param norm: definition of the used lp norm
        :return: oracle for labels
        """
        return self.compute_knn_oracle(data=data, k=1, norm=norm)

    def compute_knn_and_dists(self, data, norm=2):
        """
        Compute the nearest neighbour in the main data for each data point.
        Additionally compute the distance to this nearest neighbour and the
        distance to the nearest neighbour of another class.

        :param data: input for which the distances should be computed
        :param norm: definition of the used lp norm
        :return: nearest neighbour and two distances
        """

        data = data.to(self.device)
        raw_data, raw_labels = self._obtain_data_with_labels()

        # compute distances from each data point to each base data point
        dists = torch.cdist(data.reshape((len(data), -1)),
                            raw_data.reshape((len(self.main_data), -1)), p=norm)  # .to(self.device)

        # sort the distances + sort the base data points and their labels according to the distances w.r.t. input data
        sorted_dists, sorted_indices = dists.sort(dim=1)
        knn = raw_data[sorted_indices[:, 0]]
        knn_labels = raw_labels[sorted_indices[:, :]]

        # determine label of NN -> blow up to same shape as knn_labels data
        blow_up_l = knn_labels[:, 0].unsqueeze(dim=1).repeat_interleave(knn_labels.shape[1], dim=1).to(self.device)
        # index of first 1 = index of distance to second NN with other label in the sorted distances
        not_same_label = (blow_up_l != knn_labels) * 1
        # multiply each element in the row with decreasing number -> determine index of largest such number in each row
        # result: index of second NN with different label in sorted data
        indices = torch.argmax(not_same_label * torch.arange(not_same_label.shape[1], 0, -1, device=self.device), 1, keepdim=True)

        # get the according distances in each row
        dist_to_second_nearest = torch.gather(sorted_dists, 1, indices)

        all_dists = torch.cat((sorted_dists[:, :1], dist_to_second_nearest), dim=-1)

        return knn, all_dists

    def compute_knns(self, data, k, norm=2):
        """
        Compute the k nearest neighbours in the main data for each data point

        :param data: input for which the distances should be computed
        :param k: number of nearest neighbours to be considered
        :param norm: definition of the used lp norm
        :return: set of nearest neighbours
        """

        data = data.to(self.device)
        raw_data = self._obtain_data().to(self.device)

        dists = torch.cdist(data.reshape((len(data), -1)),
                            raw_data.reshape((len(self.main_data), -1)), p=norm)  # .to(self.device)

        _, ids = dists.topk(k, dim=1, largest=False)

        return torch.stack([raw_data[id] for id in ids])

    def compute_mean_dist(self, data, k, norm=2):
        """
        Compute the mean distance of each data point to its k nearest neighbours (w.r.t. the main data).

        :param data: input for which the distances should be computed
        :param k: number of nearest neighbours to be considered
        :param norm: definition of the used lp norm
        :return: mean distance
        """

        data = data.to(self.device)
        raw_data = self._obtain_data().to(self.device)

        dists = torch.cdist(data.reshape((len(data), -1)),
                            raw_data.reshape((len(self.main_data), -1)), p=norm)  # .to(self.device)

        dists, _ = dists.topk(k, dim=1, largest=False)

        return dists.mean(dim=1)
