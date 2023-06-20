import torch
from scipy.stats import binom_test
from statsmodels.stats.proportion import proportion_confint
from torch import nn
import numpy as np

from certified_radius import input_dependent_certified_radius_given_pb

# python certify.py toy_dataset_linear_sep models/toy/noise_0.12/checkpoint_biased_id.pth.tar 0.12 data/certify/toy/noise_0.12 --skip 1 --batch 500  --biased True
# python train.py toy_dataset_linear_sep linear_model models/toy/noise_0.12 --batch 500 --base_sigma 0.12 --epochs 25 --biased True


# https://pytorch.org/docs/stable/generated/torch.nn.Module.html
class BiasedInputDependentRSClassifier(nn.Module):
    """
    Define a biased input-dependent randomly smoothed classifier based on a chosen base classifier.
    Value of variance depends on current input variable x.
    Function defining the variance sigma based on paper: intriguing properties of input-dependent RS.
    """

    def __init__(self, base_classifier, num_classes, sigma, oracles, rate, m, device, abstain=-1):
        """
        Initialize the randomly smoothed classifier

        :param base_classifier: a base classifier
        :param num_classes: number of possible classes
        :param sigma: base value of variance
        :param oracles:
        :param rate: semi-elasticity constant for chosen sigma function
        :param m: normalization constant for data set
        :param device:
        :param abstain: value to be returned when smoothed classifier should abstain
        """

        super().__init__()

        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.oracles = oracles
        self.rate = rate
        self.m = m
        self.device = device
        self.abstain = abstain

    # currently static for testing bias function for specific case (data linearly separable)
    def bias_id(self, x_index):
        """

        :return:
        """
        orthogonal_vector = torch.tensor([-1.0, 1.0])

        return orthogonal_vector * self.oracles[x_index]

    def sigma_id(self, x_index):
        """
        Input-dependent function to compute the variance for the sampling based on the k-nearest neighbours..
        Based on function proposed in: intriguing properties of input-dependent RS

        :param x_index: index of the current point
        :return: variance w.r.t. current input
        """

        # return self.sigma * np.exp(self.rate * (self.distances[x_index] - self.m))
        return self.sigma

    def sample_under_noise(self, x, x_index, n, batch_size):
        """
        Compute the class counts of the predictions of n perturbed inputs based on the base classifier.

        :param x: current input (shape: [1, channels, image_width, image_height])
        :param x_index: index of the current point
        :param n: number of samples
        :param batch_size: size of the batches in which the evaluation should be performed
        :return: tensor containing the class counts
        """

        # create tensor containing all classes (from 0 to num_classes-1) and the according counts
        classes = torch.arange(self.num_classes).to(self.device)
        class_counts = torch.zeros([self.num_classes]).to(self.device)

        # sample and evaluate perturbed samples in batches
        remaining_samples = n
        with torch.no_grad():
            while remaining_samples > 0:
                # define suitable current batch size
                current_batch = min(batch_size, remaining_samples)
                remaining_samples -= current_batch

                # create tensor containing n times the sample x
                repeat_x_n_times = x.repeat(current_batch, 1)
                bias = self.bias_id(x_index).repeat(current_batch, 1)

                # generate and evaluate (/classify) the perturbed samples
                noise = bias + torch.randn_like(repeat_x_n_times, device=self.device) * self.sigma_id(x_index)
                perturbed_predictions = self.forward(repeat_x_n_times + noise)

                # predicted class for one sample = index of highest value
                predicted_classes = (perturbed_predictions.argmax(dim=1, keepdim=True) == classes)

                # sum over each column -> "count number of samples predicted as (column) i"
                class_counts += predicted_classes.sum(dim=0)

        return class_counts

    def predict(self, x, x_index, n, alpha, batch_size):
        """
        Evaluate the smoothed classifier g (based on the base classifier) at x.

        :param x: current input (shape: [1, channels, image_width, image_height])
        :param x_index: index of the current point
        :param n: number of samples
        :param alpha: probability that function will return a class other than g(x)
        :param batch_size: size of the batches in which the evaluation should be performed
        :return: predicted class of input
        """

        # set model to evaluation mode
        self.base_classifier.eval()

        counts = self.sample_under_noise(x, x_index, n, batch_size)

        # get values of the top two values in counts
        top_2_counts = torch.topk(counts, k=2)
        n_a, n_b = top_2_counts.values

        if binom_test(n_a, n_a + n_b, 0.5) <= alpha:
            return top_2_counts.indices[0].item()
        return self.abstain

    @staticmethod
    def lower_conf_bound(k, n, alpha):
        """
        Compute a one-sided (1-alpha) lower confidence interval using the Clopper-Pearson confidence interval.
        (See paper: certified adversarial robustness via RS)

        :param k: sample from Binomial(n,p) distribution
        :param n: parameter of the Binomial distribution
        :param alpha: parameter for the confidence interval
        :return: float representing the lower confidence bound
        """

        return proportion_confint(k, n, alpha=2 * alpha, method="beta")[0]

    def certify(self, x, x_index, n_0, n, alpha, batch_size, dim, num_steps):
        """
        Certify the robustness of the smoothed classifier g (based on f) around x

        :param x: current input (shape: [1, channels, image_width, image_height])
        :param x_index: index of the current point
        :param n_0: number of samples to estimate the class (c_a)
        :param n: number of samples to estimate a lower bound on the probability for the class (p_a)
        :param alpha: probability that function will return a class other than g(x)
        :param batch_size: size of the batches in which the evaluation should be performed
        :param dim:
        :param num_steps:
        :return: tuple of predicted class of input and according robust radius
        """

        # set model to evaluation mode
        self.base_classifier.eval()

        # generate samples and estimate most likely class
        counts_0 = self.sample_under_noise(x, x_index, n_0, batch_size)
        c_a = counts_0.argmax().item()  # .item() = get value of tensor

        # generate samples and estimate lower bound of p_a
        counts = self.sample_under_noise(x, x_index, n, batch_size)
        p_a = self.lower_conf_bound(counts[c_a].cpu(), n, alpha)

        if p_a > 0.5:
            # sigma_0 = self.sigma_id(x_index)
            # radius = input_dependent_certified_radius_given_pb(sigma_0, self.rate, dim, 1 - p_a, num_steps)
            return c_a, 0.0  # radius
        return self.abstain, 0.0

    def forward(self, x):
        """
        Classify the perturbed samples x using the base classifier.

        :param x: current input (shape: [batch_size, channels, image_width, image_height])
        :return: evaluation of the perturbed input samples
        """

        return self.base_classifier(x)