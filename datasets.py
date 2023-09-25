# function from IDRS implementation

from torchvision import transforms, datasets
from typing import *
import torch
import os
from torch.utils.data import Dataset
from toy_datasets.data_lin_separable import ToyDatasetLinearSeparationTrain, ToyDatasetLinearSeparationTest
from toy_datasets.data_cone_shaped import ToyDatasetConeShapedTrain, ToyDatasetConeShapedTest
from toy_datasets.data_blobs import ToyDatasetBlobsTrain, ToyDatasetBlobsTest

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "IMAGENET_DIR"

# list of all datasets
# DATASETS = ["imagenet", "cifar10", "unaugmented_cifar10", "mnist", "unaugmented_mnist", "toy_dataset_linear_sep"]
DATASETS = ["cifar10", "toy_dataset_linear_sep", "toy_dataset_blobs", "toy_dataset_cone_shaped"]


def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if not os.path.exists("./dataset_cache"):
        os.makedirs("./dataset_cache", exist_ok=True)

    if dataset == "imagenet":
        return _imagenet(split)
    elif dataset == "cifar10":
        return _cifar10(split)
    elif dataset == "unaugmented_cifar10":
        return _unaugmented_cifar10(split)
    elif dataset == "mnist":
        return _mnist(split)
    elif dataset == "unaugmented_mnist":
        return _unaugmented_mnist(split)
    elif dataset == "toy_dataset_linear_sep":
        return _toy_dataset_linear_sep(split)
    elif dataset == "toy_dataset_blobs":
        return _toy_dataset_blobs(split)
    elif dataset == "toy_dataset_cone_shaped":
        return _toy_dataset_cone_shaped(split)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif "blob" in dataset:
        return 4
    elif "toy_dataset" in dataset:
        return 2
    else:
        return 10


def get_normalize_layer(dataset: str, device) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV, device)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV, device)
    elif dataset == "unaugmented_cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV, device)
    elif dataset == "mnist":
        return NormalizeLayer(_MNIST_MEAN, _MNIST_STDDEV, device)
    elif dataset == "unaugmented_mnist":
        return NormalizeLayer(_MNIST_MEAN, _MNIST_STDDEV, device)


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

_MNIST_MEAN = [0.1307]
_MNIST_STDDEV = [0.3081]


def _cifar10(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())


def _unaugmented_cifar10(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.ToTensor())
    elif split == "test":
        return datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())


def _mnist(split: str) -> Dataset:
    if split == "train":  # TODO: Define the mnist_cache folders
        return datasets.MNIST("./mnist_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.MNIST("./mnist_cache", train=False, download=True, transform=transforms.ToTensor())


def _unaugmented_mnist(split: str) -> Dataset:
    if split == "train":  # TODO: Define the mnist_cache folders
        return datasets.MNIST("./mnist_cache", train=True, download=True, transform=transforms.ToTensor())
    elif split == "test":
        return datasets.MNIST("./mnist_cache", train=False, download=True, transform=transforms.ToTensor())


def _imagenet(split: str) -> Dataset:
    if IMAGENET_LOC_ENV not in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    return datasets.ImageFolder(subdir, transform)


def _toy_dataset_linear_sep(split: str) -> Dataset:
    if split == "train":
        return ToyDatasetLinearSeparationTrain(500, 1, -1)
    elif split == "test":
        return ToyDatasetLinearSeparationTest(100, 1, -1)


def _toy_dataset_blobs(split: str) -> Dataset:
    if split == "train":
        return ToyDatasetBlobsTrain(500)
    elif split == "test":
        return ToyDatasetBlobsTest(100)


def _toy_dataset_cone_shaped(split: str) -> Dataset:
    if split == "train":
        return ToyDatasetConeShapedTrain(500, 2, 5, 0.5)
    elif split == "test":
        return ToyDatasetConeShapedTest(100, 2, 5, 0.5)


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float], device):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        :param device:
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).to(device)
        self.sds = torch.tensor(sds).to(device)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds
