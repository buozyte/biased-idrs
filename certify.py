# evaluate a smoothed classifier on a dataset
import argparse
import datetime
from time import time
import os
import pickle as pkl

import torch
from torch.utils.data import DataLoader
from torch.nn import Sequential

import datasets as ds
import utils
from knn import KNNDistComp
from models.base_models.architectures import get_architecture
from models import rs
from input_dependent_functions.bias_functions import BIAS_FUNCTIONS
from input_dependent_functions.variance_functions import VARIANCE_FUNCTIONS
from constants import DATASETS

from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)
logger_file = logging.getLogger(__name__+"_file")


def main_certify(dataset, trained_classifier, base_sigma, out_dir, batch=1000, skip=1, max=-1, split="test", N0=100,
                 N=100000, alpha=0.001, norm=2, num_workers=2, use_cuda=False, index_min=0, index_max=10000,
                 id_var=False, num_nearest=20, var_func=None, rate=0.01, biased=False, num_nearest_bias=5,
                 bias_weight=1, bias_func=None, lipschitz_const=0.0, external_logger=None, add_bias_layer=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the base classifier
    checkpoint = torch.load(trained_classifier)  # if I wanna use cuda, delete the map_location argument
    base_classifier = get_architecture(checkpoint["arch"], dataset, device)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_dir = Path(out_dir)

    logfile_name = Path(out_dir) / "results.txt"
    path_results = Path(out_dir) / "results.pkl"
    logger_file.addHandler(logging.FileHandler(logfile_name))
    logger_file.info(f"idx | label | predict | radius | correct | time")


    # --- prepare data ---
    train_dataset = ds.get_dataset(dataset, "train")
    test_dataset = ds.get_dataset(dataset, "test")
    num_classes = ds.get_num_classes(dataset)

    spatial_size, num_channels, norm_const = \
        ds.get_dataset_additional_parameters(dataset)
    # --------------------

    # add bias layer if necessary
    if add_bias_layer:
        bias_layer = utils.AddSkipConnection(
            utils.ConstantBiasLayer((num_channels, spatial_size, spatial_size)))
        base_classifier = Sequential(bias_layer, base_classifier)

    # create the smoothed classifier g
    if biased:
        # ---------------------------------------------------------
        # computation of KNN related values (for variance and bias)
        knn_computer = KNNDistComp(train_dataset, num_workers, device)
        test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0, pin_memory=False)

        oracles = torch.zeros(10000)
        if num_channels == 0:
            knns = torch.zeros(10000, spatial_size)
        else:
            knns = torch.zeros(10000, num_channels, spatial_size, spatial_size)
        distances = torch.zeros(10000, 2)
        mean_distances = torch.zeros(10000)
        for i, (test_data, labels) in enumerate(test_dataloader):
            if bias_func == "mu_toy":
                oracles[i * 100:(i + 1) * 100] = knn_computer.compute_knn_oracle(test_data, k=num_nearest_bias,
                                                                                 norm=norm)
            if bias_func == "mu_knn_based":
                knns[i * 100:(i + 1) * 100, :], distances[i * 100:(i + 1) * 100, :] = knn_computer.compute_knn_and_dists(test_data, norm=norm)
            if var_func == "sigma_knn":
                mean_distances[i * 100:(i + 1) * 100] = knn_computer.compute_mean_dist(test_data, num_nearest, norm)

        distances = distances.numpy()
        mean_distances = mean_distances.numpy()
        oracles = oracles.numpy()
        if "toy" in dataset:
            oracles[oracles == 0] = -1
        # ---------------------------------------------------------

        smoothed_classifier = rs.BiasedIDRSClassifier(base_classifier=base_classifier, num_classes=num_classes,
                                                   sigma=base_sigma, device=device, bias_func=bias_func,
                                                   variance_func=var_func, oracles=oracles,
                                                   bias_weight=bias_weight, lipschitz=lipschitz_const,
                                                   knns=knns, distances=distances, rate=rate,
                                                   mean_distances=mean_distances, m=norm_const).to(device)
    elif id_var:
        # obtain knn distances of test data
        dist_computer = KNNDistComp(train_dataset, num_workers, device)
        test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False,
                                     num_workers=0, pin_memory=False)
        distances = torch.zeros(10000)
        for i, (test_data, labels) in enumerate(test_dataloader):
            distances[i * 100:(i + 1) * 100] = dist_computer.compute_mean_dist(test_data, num_nearest, norm)
        distances = distances.numpy()

        smoothed_classifier = rs.IDRSClassifier(base_classifier=base_classifier, num_classes=num_classes,
                                             sigma=base_sigma, distances=distances, rate=rate, m=norm_const,
                                             device=device).to(device)

    else:
        smoothed_classifier = rs.RSClassifier(base_classifier=base_classifier, num_classes=num_classes,
                                           sigma=base_sigma, device=device).to(device)

    smoothed_classifier.load_state_dict(checkpoint['state_dict'])
    # TODO:
    #   1. use real alternative classifiers
    #   2. maybe find a nicer way to set/pass the alternative classifier
    #   3. handle possibility to choose different alternative classifiers for (each) dataset -> via if cases
    if biased and bias_func is not None and "gradient" in bias_func:
        # just for testing :)
        alt_classifier = rs.BiasedIDRSClassifier(base_classifier=base_classifier, num_classes=num_classes,
                                                      sigma=0.25, device=device).to(device)
        alt_classifier.load_state_dict(checkpoint['state_dict'])
        smoothed_classifier.alt_classifier = alt_classifier

    if dataset == "toy_dataset_linear_sep":
        if not biased:
            oracles = None
            knns = None
            distances = None
        if bias_func == "mu_toy":
            train_dataset.visualize_with_classifier_oracle_based(smoothed_classifier, bias_weight=bias_weight,
                                                                 file_path=out_dir)
            test_dataset.visualize_with_classifier_oracles_based(smoothed_classifier, bias_weight=bias_weight,
                                                                 oracles=oracles, file_path=out_dir)
        elif bias_func == "mu_knn_based":
            train_dataset.visualize_with_classifier_knn_based(smoothed_classifier, bias_weight=bias_weight,
                                                              file_path=out_dir)
            test_dataset.visualize_with_classifier_knn_based(smoothed_classifier, bias_weight=bias_weight,
                                                             knns=knns, distances=distances, file_path=out_dir)

    correct_sum = 0
    certified_sum = 0
    overall_time_start = time()
    results = {
        "sample_index": [],
        "correct": [],
        "certified_radius": []
    }
    for i in range(index_min, index_max):
        # only certify every skip examples, and stop after max examples
        if i % skip != 0:
            continue
        if i == max:
            break

        (inputs, label) = test_dataset[i]

        before_time = time()
        # certify the prediction of g around x
        inputs = inputs.to(device)

        dim = torch.prod(torch.tensor(inputs.shape))
        if id_var or biased:
            prediction, radius = smoothed_classifier.certify(inputs, i, N0, N, alpha, batch, dim.item(), 1000)
        else:
            prediction, radius = smoothed_classifier.certify(inputs, N0, N, base_sigma, alpha, batch)
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = after_time - before_time  # 
        logger_file.info(
            f"{i:<4}| " + \
            f"{label:<6}| " + \
            f"{prediction:<8}| " + \
            f"{radius:<7.3}| " + \
            f"{correct:<8}| " + \
            f"{str(datetime.timedelta(seconds=time_elapsed)):<11}")

        correct_sum += correct
        certified_sum += 1

        utils.dict_append(results,
            {
                "sample_index": i,
                "correct": correct,
                "certified_radius": radius}
        )

        if external_logger is not None:
            current_result = {
                "sample_index": i,
                "num_samples": certified_sum,
                "true_label": label,
                "predicted_label": prediction,
                "correct": {
                    "current_sample": correct,
                    "overall": correct_sum,
                },
                "certified_radius": radius,
                "time": {
                    "elapsed": time_elapsed,
                    "overall": after_time - overall_time_start,
                }
            }
            external_logger(current_result)


    with open(path_results, "wb") as f:
        pkl.dump(results, f)
    logger_file.info(f"\nTotal correct / total certified: {correct_sum} / {certified_sum}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Certify many examples non-constant sigma')
    parser.add_argument("dataset", choices=DATASETS,
                        help="which dataset")
    parser.add_argument("trained_classifier", type=str,
                        help="path to saved pytorch model of the trained classifier")
    parser.add_argument("base_sigma", type=float, default=0.5,
                        help="base smoothing strength for samples closest to the boundary")
    parser.add_argument("out_dir", type=str,
                        help="output directory")
    parser.add_argument("--batch", type=int, default=1000,
                        help="batch size")
    parser.add_argument("--skip", type=int, default=1,
                        help="how many examples to skip")
    parser.add_argument("--max", type=int, default=-1,
                        help="stop after this many examples")
    parser.add_argument("--split", choices=["train", "test"], default="test",
                        help="train or test set")
    parser.add_argument("--N0", type=int, default=100)
    parser.add_argument("--N", type=int, default=100000,
                        help="number of samples to use")
    parser.add_argument("--alpha", type=float, default=0.001,
                        help="failure probability")
    parser.add_argument("--norm", type=int, default=2,
                        help="norm to be used in KNN")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of workers in dataloader")
    parser.add_argument("--use_cuda", type=bool, default=False,
                        help="whether to use cuda")
    parser.add_argument("--index_min", type=int, default=0,
                        help="Min index of sample to evaluate")
    parser.add_argument("--index_max", type=int, default=10000,
                        help="Max index of sample to evaluate")
    # variance
    parser.add_argument('--id_var', default=False, type=bool,
                        help="Indicator whether to use input-dependent computation of the variance")
    parser.add_argument('--num_nearest', default=20, type=int,
                        help='How many nearest neighbors to use')
    parser.add_argument('--var_func', default=None, type=str, choices=VARIANCE_FUNCTIONS,
                        help='Choice for the variance function to be used')
    parser.add_argument("--rate", type=float, default=0.01,
                        help="exponential rate of increase in sigma")
    # bias
    parser.add_argument('--biased', default=False, type=bool,
                        help="Indicator whether to use a biased")
    parser.add_argument('--num_nearest_bias', default=5, type=int,
                        help='How many nearest neighbors to use')
    parser.add_argument('--bias_weight', default=1, type=float,
                        help="Weight of bias")
    parser.add_argument('--bias_func', default=None, type=str, choices=BIAS_FUNCTIONS,
                        help='Choice for the bias function to be used')
    parser.add_argument('--lipschitz_const', default=0.0, type=float,
                        help="Lipschitz constant of the bias functions")
    parser.add_argument('--add_bias_layer', type=bool, default=False)
    args = parser.parse_args()

    args_dict = vars(args)

    main_certify(**args_dict)
