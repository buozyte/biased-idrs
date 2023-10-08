import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from sklearn.datasets import make_gaussian_quantiles
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import logging
logger = logging.getLogger(__name__)

device = torch.device("cuda")



def gaussian_normalization(inputs, sigmas, num_channels, spatial_size, device):
    """
    Normalization methods for the input data.
    (Note: very time consuming)

    :param inputs: inputs to be normalized
    :param sigmas: set of input-dependent sigmas for each input
    :param num_channels: number of channels (in picture)
    :param spatial_size: spatial size of input (i.e. height/width of picture)
    :param device: device for device handling
    :return: normalized inputs
    """

    sigmas_2d = sigmas.unsqueeze(dim=1).repeat_interleave(num_channels, dim=1).to(device)
    sigmas_3d = sigmas_2d.unsqueeze(dim=2).repeat_interleave(spatial_size, dim=2).to(device)
    sigmas_image_size = sigmas_3d.unsqueeze(dim=3).repeat_interleave(spatial_size, dim=3).to(device)

    noisy_inputs = inputs + torch.randn_like(inputs, device=device) * sigmas_image_size
    return noisy_inputs


class AverageMeter(object):
    """
    Compute and store the average and current value.
    """
    
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """
        Reset all values to 0.
        """

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the average value based on the new value

        :param val: new value
        :param n: number of times newest value is added
        """
        
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def toy_accuracy(output, target, topk=(1,)):
    """
    Compute the accuracy over the k top predictions for the specified values of k for the toy example.
    """
    
    with torch.no_grad():
        batch_size = target.size(0)

        pred = output.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy(output, target, topk=(1,)):
    """
    Compute the accuracy over the k top predictions for the specified values of k.
    """
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def init_logfile(filename: str, text: str):
    """
    Initialize log file and add given text.

    :param filename: filename for the logfile
    :param text: text to be added
    """
    
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()


def log(filename: str, text: str):
    """
    Add given text to logfile.

    :param filename: filename of the logfile
    :param text: text to be added
    """

    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()


def get_toy_dataset_2d(option="gaussian", bounds=(0, 1), **kwargs):
    """ Creates a 2d dataset with 2 classes. Available options are 'gaussian' and 'random'.

    """

    # two gaussians
    if option == "gaussian":

        # default values
        kwargs_ = {
            "class1": {
                "mean": (0, 0),
                "cov": 3,
                "n_samples": 2500
            },
            "class2": {
                "mean": (4, 4),
                "cov": 1,
                "n_samples": 3000
            }
        }

        if kwargs:
            kwargs_["class1"].update(kwargs["class1"])
        X1, y1 = make_gaussian_quantiles(n_classes=2, n_features=2, random_state=1, **kwargs_["class1"])
        if kwargs:
            kwargs_["class2"].update(kwargs["class2"])
        X2, y2 = make_gaussian_quantiles(n_classes=2, n_features=2, random_state=1, **kwargs_["class2"])

        # Combine the gaussians
        X = torch.Tensor(np.concatenate((X1, X2)))
        y = torch.LongTensor(np.concatenate((y1, - y2 + 1)))

    # random points at least 2r apart
    elif option == "random":
        # default values
        kwargs_ = {
            "N": 20, # number of data points
            "r": 0.1, # min radius between each 2 points
            "seed": 1234
        }

        if kwargs:
            kwargs_.update(kwargs)
        np.random.seed(kwargs_["seed"])
        x = [np.random.uniform(size=(2))]
        while(len(x) < kwargs_["N"]):
            p = np.random.uniform(size=(2))
            if min(np.abs(p-a).sum() for a in x) > 2*kwargs_["r"]:
                x.append(p)

        X = torch.Tensor(np.array(x))
        #torch.manual_seed(1)
        y = (torch.rand(kwargs_["N"]) + 0.5).long()


    if bounds:
        X = (X + (bounds[0] - X.min())) / (X.max() - X.min()) * (bounds[1] - bounds[0])

    return X.to(device), y.to(device)


def plot_network_boundary_2d(network, a0, a1, N, x_base=None, r=None, data=False, scale=None):
    """ Plots the decision regions of a network (2 features, 2 classes) in a given region.

    Input:
    ------
    network - object implementing the forward pass through the network
    a0, a1 - tuples, lower left and upper right corners of the region to plot
    N - number of points in the mesh (same for both 2 dimentions)

    """
    XX, YY = np.meshgrid(np.linspace(a0[0], a1[0], N), np.linspace(a0[1], a1[1], N))
    X0 = torch.autograd.Variable(torch.Tensor(np.stack([np.ravel(XX), np.ravel(YY)]).T).to(device))

    # if scale is given rescale the inputs before applying the network
    if not scale:
        y0 = network(X0)
    else:
        y0 = network((X0-scale["mean"]) / scale["std"])

    ZZ = (y0[:,0] - y0[:,1]).resize(N, N).cpu().data.numpy()

    fig, ax = plt.subplots(figsize=(8,8))
    ax.contourf(XX, YY, -ZZ, cmap="coolwarm", levels=np.array((-1e6, 0.0, 1e6)))

    # plot the datapoints if given data = (X, y)
    if data:
        X, y = data
        ax.scatter(X.cpu().numpy()[:,0], X.cpu().numpy()[:,1], c=y.cpu().numpy(), cmap="coolwarm", s=2)

    ax.axis("equal")
    ax.axis([a0[0],a1[0],a0[1],a1[1]])

    if (x_base is not None) and (r is not None):
        for x_base_, r_ in zip(x_base, r):
            ax.add_patch(patches.Circle(x_base_, r_, fill=False))


def get_dense_network(n_layers, n_neurons, n_in=2, n_out=2):
    """ Creates a dense ReLU network with the same number of neurons in each hidden layer.

    """
    layers = sum(
        [[nn.Linear(n_neurons, n_neurons), nn.ReLU()] for _ in range(n_layers-1)],
        [nn.Linear(n_in, n_neurons), nn.ReLU()]
    )
    layers.append(nn.Linear(n_neurons,n_out))

    network = nn.Sequential(*layers   )
    return network


def train(network, X, y, lr=1e-3, epochs=1000, opt=None):
    """ Trains a network and its substitue, then plots their boundaries.

    """
    opt = opt if opt is not None else optim.Adam(network.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    for epoch in range(1, epochs+1):
        out = network(X)#(Variable(X))
        l = loss(out, y)#Variable(y))
        err = (out.max(1)[1].data != y).float().mean()
        if epoch % int(epochs/10) == 0:
            logger.info(f"loss: {l.item():.4f}, error: {err:.6f}, epoch: {epoch:4}/{epochs}")

        opt.zero_grad()
        l.backward()
        opt.step()