import torch
from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn

from models.base_models.cifar_resnet import resnet_cifar, resnet_mnist
from models.base_models.linear_classifier import LinearModel
from models.base_models.toy_model import ToyModel
from datasets import get_normalize_layer

# resnet50 - the classic ResNet-50, sized for ImageNet
# mnist_resnet20 - a 20-layer residual network sized for MNIST
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["resnet50", "cifar_resnet20", "cifar_resnet110", "mnist_resnet20", "toy_model", "linear_model"]


def get_architecture(arch: str, dataset: str, device) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :param device:
    :return: a Pytorch module
    """
    if arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=False)).to(device)
        cudnn.benchmark = True
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10).to(device)
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10).to(device)
    elif arch == "mnist_resnet20":
        model = resnet_mnist(depth=20, num_classes=10).to(device)
    elif arch == "linear_model":
        model = LinearModel(input_dim=2, num_classes=2)
        return model
    elif arch == "toy_model":
        model = ToyModel(input_dim=2, num_classes=2)
        return model
    normalize_layer = get_normalize_layer(dataset, device)
    return torch.nn.Sequential(normalize_layer, model)
