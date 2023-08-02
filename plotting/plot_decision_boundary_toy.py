import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from datasets import get_dataset, get_num_classes
from knn import KNNDistComp
from models.base_models.architectures import get_architecture
from models.biased_idrs import BiasedIDRSClassifier
from models.input_dependent_rs import IDRSClassifier
from models.rs import RSClassifier


parser = argparse.ArgumentParser(description='Plot')
parser.add_argument('trained_model', type=str)
parser.add_argument('filepath', type=str)
parser.add_argument('filename', type=str)
parser.add_argument("base_sigma", type=float, default=0.5,
                    help="base smoothing strength for samples closest to the boundary")
parser.add_argument('--input_dependent', default=False, type=bool,
                    help="Indicator whether to use input-dependent computation of the variance")
parser.add_argument('--num_nearest_var', default=20, type=int,
                    help='How many nearest neighbors to use')
parser.add_argument("--rate", type=float, default=0.01,
                    help="exponential rate of increase in sigma")
parser.add_argument('--biased', default=False, type=bool,
                    help="Indicator whether to use a biased")
parser.add_argument('--bias_weight', default=0, type=float,
                    help="Weight of bias")
parser.add_argument('--num_nearest_bias', default=5, type=int,
                    help='How many nearest neighbors to use')
args = parser.parse_args()


# TODO: add possibility to use different functions for the bias and variance
def main():
    """
    Plot the decision boundary for the toy example.

    (Can easily be adapted to fit other examples. Only restriction: dimension of input data space = 2.)
    """

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # init base classifier
    checkpoint = torch.load(args.trained_model)
    base_classifier = get_architecture(checkpoint["arch"], "toy_dataset_linear_sep", device)

    # init KNN computer
    train_dataset = get_dataset("toy_dataset_linear_sep", "train")
    num_classes = get_num_classes("toy_dataset_linear_sep")
    knn_computer = KNNDistComp(train_dataset, 2, device)

    # init test/input data (as set of pairs (x,y))
    ls = np.linspace(-2, 2, num=200)
    xx, yy = np.meshgrid(ls, ls)
    inputs = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    input_dataloader = DataLoader(inputs, batch_size=100, shuffle=False, num_workers=0, pin_memory=False)

    # init smoothed classifier based on input
    if args.input_dependent:
        distances = torch.zeros(40000)
        for i, (test_data, labels) in enumerate(input_dataloader):
            distances[i * 100:(i + 1) * 100] = knn_computer.compute_dist(test_data, args.num_nearest_var, 2)
        distances = distances.numpy()

        smoothed_classifier = IDRSClassifier(base_classifier=base_classifier, num_classes=num_classes,
                                             sigma=args.base_sigma, distances=distances, rate=args.rate, m=1,
                                             device=device).to(device)
    elif args.biased:
        oracles = torch.zeros(40000)
        for i, data in enumerate(input_dataloader):
            oracles[i * 100:(i + 1) * 100] = knn_computer.compute_knn_oracle(data, k=args.num_nearest_bias, norm=2)
        oracles = oracles.numpy()
        oracles[oracles == 0] = -1

        smoothed_classifier = BiasedIDRSClassifier(base_classifier=base_classifier, num_classes=num_classes,
                                                   sigma=args.base_sigma, bias_weight=args.bias_weight, oracles=oracles,
                                                   device=device).to(device)

    else:
        smoothed_classifier = RSClassifier(base_classifier=base_classifier, num_classes=num_classes,
                                           sigma=args.base_sigma, device=device).to(device)
    smoothed_classifier.load_state_dict(checkpoint['state_dict'])

    outputs_0 = torch.zeros(40000)
    for i in range(40000):
        outputs_0[i] = smoothed_classifier.predict(inputs[i], i, 1000, 0.001, 100)

    c = np.array([[1.0, 1.0, 1.0, 1.0],
                  [214 / 255, 39 / 255, 40 / 255, 1.0],
                  [31 / 255, 119 / 255, 180 / 255, 1.0], ])

    plt.contourf(xx, yy, outputs_0.reshape(xx.shape), cmap=ListedColormap(c), alpha=0.5)
    plt.title(f"Decision boundary of smoothed classifier")
    plt.savefig(f"{args.path}/decision_boundary.pdf")


if __name__ == "__main__":
    main()
