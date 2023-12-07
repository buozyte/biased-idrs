import torch
from scipy.stats import binom_test, norm
from statsmodels.stats.proportion import proportion_confint
from torch import nn
import numpy as np

from input_dependent_functions import bias_functions as bf
from input_dependent_functions import variance_functions as vf

from utils import AddSkipConnection
from certified_radius import *


# https://pytorch.org/docs/stable/generated/torch.nn.Module.html
class RSClassifier(nn.Module):
    """
    Define a randomly smoothed classifier based on a chosen base classifier.
    """

    def __init__(self, base_classifier, num_classes, sigma, device, abstain=-1):
        """
        Initialize the randomly smoothed classifier
        
        :param base_classifier: a base classifier
        :param num_classes: number of possible classes
        :param sigma: parameter used to define the variance in a normal distribution
        :param device: pytorch device handling
        :param abstain: value to be returned when smoothed classifier should abstain
        """

        super().__init__()

        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.device = device
        self.abstain = abstain


    def sample_under_noise(self, x, n, sigma, batch_size=None, bias=None):
        """
        Compute the class counts of the predictions of n perturbed inputs based on the base classifier.
        
        :param model: callable object implementing the forward pass
        :param x: current input (shape: [1, channels, image_width, image_height])
        :param n: number of samples
        :param sigma: smoothing magnitude
        :param device: device for the computation

        :return: tensor containing the class counts
        """
        batch_size = batch_size if batch_size is not None else n

        # create tensor containing all classes (from 0 to num_classes-1)
        classes = torch.arange(self.num_classes).to(self.device)
        class_counts = torch.zeros([self.num_classes]).to(self.device)

        # sample and evaluate perturbed samples in batches
        remaining_samples = n
        with torch.no_grad():
            while remaining_samples > 0:
                # define suitable current batch size
                current_batch = min(batch_size, remaining_samples)
                remaining_samples -= current_batch

                rep_shape = [current_batch] + [1]*x.ndim

                # create tensor containing current_batch times the sample x
                repeat_x_n_times = x.repeat(rep_shape)
                if bias is None:
                    bias_repeat = torch.zeros_like(repeat_x_n_times)
                else:
                    bias_repeat = bias.repeat(rep_shape)

                # generate and evaluate (/classify) the perturbed samples
                noise = bias_repeat + torch.randn_like(repeat_x_n_times, device=self.device) * sigma
                perturbed_predictions = self(
                    (repeat_x_n_times + noise)#.reshape(-1, *repeat_x_n_times.shape[2:])
                    )

                # predicted class for one sample = index of highest value
                predicted_classes = (perturbed_predictions.argmax(dim=1, keepdim=True) == classes)

                # sum over each column -> "count number of samples predicted as (column) i"
                class_counts += predicted_classes.sum(dim=0)

        return class_counts


    def predict(self, x, n, sigma, alpha, batch_size=None, bias=None, return_all_counts=False):
        """
        Evaluate the smoothed classifier g (based on the base classifier) at x.
        
        :param x: current input (shape: [1, channels, image_width, image_height])
        :param n: number of samples
        :param sigma: smoothing magnitude
        :param alpha: probability that function will return a class other than g(x)
        :param batch_size: size of the batches in which the evaluation should be performed

        :return: predicted class of input
        """

        batch_size = batch_size if batch_size is not None else n

        # set model to evaluation mode
        self.base_classifier.eval()

        counts = self.sample_under_noise(x, n, sigma, batch_size, bias)
        if return_all_counts:
            return counts

        # get indices and values of the top two values in counts
        top_2_counts = torch.topk(counts, k=2)
        # c_a, c_b = top_2_counts.indices
        n_a, n_b = top_2_counts.values

        if binom_test(n_a, n_a + n_b, 0.5) <= alpha:
            # return c_a
            return top_2_counts.indices[0].item()
        return self.abstain

    @classmethod
    def lower_conf_bound(cls, k, n, alpha):
        """
        Compute a one-sided (1-alpha) lower confidence interval using the Clopper-Pearson confidence interval.
        (See paper: certified adversarial robustness via RS)

        :param k: sample from Binomial(n,p) distribution
        :param n: parameter of the Binomial distribution
        :param alpha: parameter for the confidence interval
        :return: float representing the lower confidence bound
        """

        return proportion_confint(k, n, alpha=2 * alpha, method="beta")[0]


    def certify(self, x, n_sampling, n_bound, sigma, alpha, batch_size=None):
        """
        Certify the robustness of the smoothed classifier g (based on f) around x
        
        :param x: current input (shape: [1, channels, image_width, image_height])
        :param n_0: number of samples to estimate the class (c_a)
        :param n: number of samples to estimate a lower bound on the probability for the class (p_a)
        :param alpha: probability that function will return a class other than g(x)
        :param batch_size: size of the batches in which the evaluation should be performed
        :return: tuple of predicted class of input and according robust radius
        """

        # set model to evaluation mode
        self.base_classifier.eval()

        # generate samples to estimate/guess c_a
        counts_sampling = self.sample_under_noise(x, n_sampling, sigma, batch_size)
        # determine most likely class
        c_a = counts_sampling.argmax().item()  # .item() = get value of tensor

        # generate samples to estimate/guess lower bound of p_a
        counts_bound = self.sample_under_noise(x, n_bound, sigma, batch_size)
        p_a = RSClassifier.lower_conf_bound(counts_bound[c_a].cpu(), n_bound, alpha)

        if p_a > 0.5:
            return c_a, self.sigma * norm.ppf(p_a)  # norm.ppf = Phi^(-1)
        return self.abstain, 0.0


    def forward(self, x):
        """
        Generate and classify the perturbed samples x using the base classifier.
        
        :param x: current input (shape: [batch_size, channels, image_width, image_height])
        :return: evaluation of the perturbed input samples
        """

        return self.base_classifier(x)


    def add_module(self, module):
        """
        Add new module F to the base classifier f, such that the whole new forward
        pass of x is f(F(x)).

        :param module: nn.Module F
        :return: None, the self.base_classifier attribute is updated.
        """

        self.base_classifier = nn.Sequential(module, self.base_classifier)

        return




class IDRSClassifier(RSClassifier):
    """
    Define a input-dependent randomly smoothed classifier based on a chosen base classifier.
    Value of variance depends on current input variable x.
    Function defining the variance sigma based on paper: intriguing properties of input-dependent RS.
    """

    def __init__(
            self, base_classifier, num_classes, sigma, distances, rate, m, device, abstain=-1,
            mean_distances_fcn=None):
        """
        Initialize the randomly smoothed classifier

        :param base_classifier: a base classifier
        :param num_classes: number of possible classes
        :param sigma: base value of variance
        :param distances: mean distances for every sample to its k nearest neighbours
        :param rate: semi-elasticity constant for chosen sigma function
        :param m: normalization constant for data set
        :param device: device for device handling
        :param abstain: value to be returned when smoothed classifier should abstain
        """

        super().__init__(base_classifier, num_classes, sigma, device, abstain)

        self.distances = distances
        self.mean_distances_fcn = mean_distances_fcn
        self.rate = rate
        self.m = m


    def sigma_id(self, x_id):
        """
        Input-dependent function to compute the variance for the sampling based
        on the k-nearest neighbours. Based on function proposed in: intriguing
        properties of input-dependent RS

        :param x_id: index of the current point
        :return: variance w.r.t. current input
        """

        return self.sigma * np.exp(self.rate * (self.distances[x_id] - self.m))


    def sigma_fcn(self, x):
        """
        Input-dependent function to compute the variance for the sampling based
        on the k-nearest neighbours. Based on function proposed in: intriguing
        properties of input-dependent RS

        :param x: the current point
        :return: variance w.r.t. current input
        """

        return self.sigma * torch.exp(self.rate * (self.mean_distances_fcn(x) - self.m))


    def certify(self, x, x_id, n_0, n, alpha, batch_size, dim, num_steps):
        """
        Certify the robustness of the smoothed classifier g (based on f) around x

        :param x: current input (shape: [1, channels, image_width, image_height])
        :param x_index: index of the current point
        :param n_0: number of samples to estimate the class (c_a)
        :param n: number of samples to estimate a lower bound on the probability for the class (p_a)
        :param alpha: probability that function will return a class other than g(x)
        :param batch_size: size of the batches in which the evaluation should be performed
        :param dim: dimension of input data space
        :param num_steps: number of steps to take in computation of certified radius
        :return: tuple of predicted class of input and according robust radius
        """
        if x_id >= 0:
            sigma = self.sigma_id(x_id)
        else:
            sigma = self.sigma_fcn(x)

        # set model to evaluation mode
        self.base_classifier.eval()

        # generate samples and estimate most likely class
        counts_0 = self.sample_under_noise(x, n_0, sigma, batch_size)
        c_a = counts_0.argmax().item()  # .item() = get value of tensor

        # generate samples and estimate lower bound of p_a
        counts = self.sample_under_noise(x, n, sigma, batch_size)
        p_a = self.lower_conf_bound(counts[c_a].cpu(), n, alpha)

        if p_a > 0.5:
            radius = input_dependent_certified_radius_given_pb(sigma, self.rate, dim, 1-p_a, num_steps)
            return c_a, radius
        return self.abstain, 0.0




class BiasedIDRSClassifier(RSClassifier):
    """
    Define a biased input-dependent randomly smoothed classifier based on a chosen base classifier.
    Value of variance depends on current input variable x.
    Function defining the variance sigma based on paper: intriguing properties of input-dependent RS.
    """

    def __init__(self, base_classifier, num_classes, sigma, device,
                 bias_func=None, variance_func=None, oracles=None,
                 bias_weight=1, lipschitz=0, knns=None, distances=None,
                 rate=0, mean_distances=None, m=0, alt_classifier=None,
                 knns_fcn=None, distances_fcn=None, mean_distances_fcn=None,
                 abstain=-1):
        """
        Initialize the randomly smoothed classifier

        :param base_classifier: a base classifier
        :param num_classes: number of possible classes
        :param sigma: base value of variance
        :param device: device for device handling
        :param oracles: output oracle for each sample based on the k nearest neighbours
        :param bias_weight: "weight" of the bias
        :param lipschitz: Lipschitz constant for the chosen mu function
        :param knns: k-nearest neighbours for each sample
        :param distances: according distances for every sample to its k-nearest neighbours
        :param rate: semi-elasticity constant for chosen sigma function
        :param mean_distances: mean of distances for every sample to its k-nearest neighbours
        :param m: normalization constant for data set
        :param alt_classifier: pre-trained alternative classifier for the current task
        :param abstain: value to be returned when smoothed classifier should abstain
        """

        super().__init__(base_classifier, num_classes, sigma, device, abstain)


        self.bias_func = bias_func
        self.biased_cr = (bias_func is not None)
        self.variance_func = variance_func
        self.var_id_cr = (variance_func is not None)
        self.oracles = oracles
        self.bias_weight = bias_weight
        self.lipschitz = lipschitz
        self.distances = distances
        self.knns = knns
        self.rate = rate
        self.mean_distances = mean_distances
        self.m = m
        self.alt_classifier = alt_classifier

        self.distances_fcn = distances_fcn
        self.knns_fcn = knns_fcn
        self.mean_distances_fcn = mean_distances_fcn





    def bias_id(self, x, x_index):
        """
        Compute the bias based on the chosen bias function and the according parameters.

        :param x: current sample/point
        :param x_index: index of the current point
        :return: bias w.r.t. current input
        """
        
        if self.bias_func is None or self.bias_func == "":
            return torch.zeros_like(x)
        
        if self.bias_func == "mu_toy":
            return bf.mu_toy(self.oracles, x_index, self.base_classifier, self.device)
        if self.bias_func == "mu_knn_based":
            return bf.mu_nearest_neighbour(x, x_index, self.knns, self.distances, self.device)
        if self.bias_func == "mu_gradient_based":
            return bf.mu_gradient(self.alt_classifier, x, self.device)

        return torch.zeros_like(x)


    def bias_fcn(self, x):
        """
        Compute the bias based on the chosen bias function and the according parameters.

        :param x: current sample/point
        :return: bias w.r.t. current input
        """

        if self.bias_func == "mu_knn_based":
            return bf.mu_nearest_neighbour_fcn(x, self.knns_fcn, self.distances_fcn, self.device)
        
        else:
            raise ValueError(f"Only 'mu_knn_based' bias is supported, got '{self.bias_func}'.")


    def sigma_id(self, x_index):
        """
        Compute the variance based on the chosen variance function and the according parameters.

        :param x_index: index of the current point
        :return: variance w.r.t. current input
        """

        if self.variance_func is None or self.variance_func == "":
            return self.sigma
        
        if self.variance_func == "sigma_knn":
            return vf.sigma_knn(self.sigma, self.rate, self.m, self.mean_distances, x_index)

        return self.sigma


    def sigma_fcn(self, x):
        """
        Compute the variance based on the chosen variance function and the according parameters.

        :param x_index: the current point
        :return: variance w.r.t. current input
        """

        if self.variance_func == "sigma_knn":
            return vf.sigma_knn_fcn(self.sigma, self.rate, self.m, self.mean_distances_fcn, x)

        else:
            raise ValueError(f"Only 'sigma_knn' variance is supported, got '{self.variance_func}'.")



    def certify(self, x, x_id, n_0, n, alpha, batch_size, dim, num_steps):
        """
        Certify the robustness of the smoothed classifier g (based on f) around x

        :param x: current input (shape: [1, channels, image_width, image_height])
        :param x_index: index of the current point
        :param n_0: number of samples to estimate the class (c_a)
        :param n: number of samples to estimate a lower bound on the probability for the class (p_a)
        :param alpha: probability that function will return a class other than g(x)
        :param batch_size: size of the batches in which the evaluation should be performed
        :param dim: dimension of input data space
        :param num_steps: number of steps to take in computation of certified radius
        :return: tuple of predicted class of input and according robust radius
        """
        if x_id >= 0:
            bias = self.bias_id(x, x_id) * self.bias_weight
        else:
            bias = self.bias_fcn(x) * self.bias_weight

        # set model to evaluation mode
        self.base_classifier.eval()

        # generate samples and estimate most likely class
        counts_0 = self.sample_under_noise(x, n_0, batch_size, bias)
        c_a = counts_0.argmax().item()  # .item() = get value of tensor

        # generate samples and estimate lower bound of p_a

        counts = self.sample_under_noise(x, n, batch_size, bias)
        p_a = self.lower_conf_bound(counts[c_a].cpu(), n, alpha)

        if p_a > 0.5:
            sigma_0 = self.sigma_id(x_id)

            if sigma_0 == 0:
                return c_a, 0

            if self.biased_cr:
                radius = biased_input_dependent_certified_radius_given_pb(sigma_0, self.lipschitz, self.rate, dim,
                                                                          1 - p_a, num_steps)

            elif self.var_id_cr:
                radius = input_dependent_certified_radius_given_pb(sigma_0, self.rate, dim, 1 - p_a, num_steps)

            else:
                radius = certified_radius_given_pa(sigma_0, p_a)

            return c_a, radius

        return self.abstain, 0.0