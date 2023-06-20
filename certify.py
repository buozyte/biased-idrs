# evaluate a smoothed classifier on a dataset
import argparse
import datetime
from time import time
import os

import torch
from torch.utils.data import DataLoader

from datasets import get_dataset, DATASETS, get_num_classes
from knn import KNNDistComp
from models.architectures import get_architecture
from models.input_dependent_rs import InputDependentRSClassifier
from models.random_smooth import RandSmoothedClassifier
from models.biased_idrs import BiasedInputDependentRSClassifier

parser = argparse.ArgumentParser(description='Certify many examples non-constant sigma')
parser.add_argument("dataset", choices=DATASETS,
                    help="which dataset")
parser.add_argument("base_classifier", type=str,
                    help="path to saved pytorch model of base classifier")
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
parser.add_argument("--rate", type=float, default=0.01,
                    help="exponential rate of increase in sigma")
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
parser.add_argument('--input_dependent', default=False, type=bool,
                    help="Indicator whether to use input-dependent computation of the variance")
parser.add_argument('--num_nearest', default=20, type=int,
                    help='How many nearest neighbors to use')
parser.add_argument('--biased', default=False, type=bool,
                    help="Indicator whether to use a biased")
args = parser.parse_args()


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if args.input_dependent:
        add_model_type = "_id"
    elif args.biased:
        add_model_type = "_biased_id"
    else:
        add_model_type = ""

    # load the base classifier
    checkpoint = torch.load(args.base_classifier)  # if I wanna use cuda, delete the map_location argument
    base_classifier = get_architecture(checkpoint["arch"], args.dataset, device)

    # prepare output file
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    outfile = os.path.join(args.out_dir, f'sigma_{args.base_sigma}{add_model_type}')
    f = open(outfile, 'a')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # --- prepare data ---
    train_dataset = get_dataset(args.dataset, "train")
    test_dataset = get_dataset(args.dataset, "test")

    if args.dataset == 'cifar10':
        norm_const = 5
    elif args.dataset == 'mnist':
        norm_const = 1.5
    else:
        norm_const = 0
    # --------------------

    # create the smoothed classifier g
    num_classes = get_num_classes(args.dataset)
    if args.input_dependent:
        # obtain knn distances of test data
        dist_computer = KNNDistComp(train_dataset, args.num_workers, device)
        test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False,
                                     num_workers=0, pin_memory=False)
        distances = torch.zeros(10000)
        for i, (test_data, labels) in enumerate(test_dataloader):
            distances[i * 100:(i + 1) * 100] = dist_computer.compute_dist(test_data, 20, args.norm)
        distances = distances.numpy()
    
        smoothed_classifier = InputDependentRSClassifier(base_classifier=base_classifier, num_classes=num_classes,
                                                         sigma=args.base_sigma, distances=distances, rate=args.rate,
                                                         m=norm_const, device=device).to(device)
    elif args.biased:
        # obtain knn distances of test data
        knn_computer = KNNDistComp(train_dataset, args.num_workers, device)
        test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False,
                                     num_workers=0, pin_memory=False)
        oracles = torch.zeros(10000)
        for i, (test_data, labels) in enumerate(test_dataloader):
            oracles[i * 100:(i + 1) * 100] = knn_computer.compute_1nn_oracle(test_data, args.norm)
        oracles = oracles.numpy()
        oracles[oracles == 0] = -1

        smoothed_classifier = BiasedInputDependentRSClassifier(base_classifier=base_classifier, num_classes=num_classes,
                                                               sigma=args.base_sigma, oracles=oracles, rate=args.rate,
                                                               m=norm_const, device=device).to(device)

    else:
        smoothed_classifier = RandSmoothedClassifier(base_classifier=base_classifier, num_classes=num_classes,
                                                     sigma=args.base_sigma, device=device).to(device)

    smoothed_classifier.load_state_dict(checkpoint['state_dict'])

    if "toy" in args.dataset:
        train_dataset.visualize_with_classifier(smoothed_classifier, save=False, show=True)
        test_dataset.visualize_with_classifier(smoothed_classifier, save=False, show=True)

    for i in range(args.index_min, args.index_max):
        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (inputs, label) = test_dataset[i]

        before_time = time()
        # certify the prediction of g around x
        inputs = inputs.to(device)

        if args.input_dependent:
            dim = inputs.shape[0] * inputs.shape[1] * inputs.shape[2]
            prediction, radius = smoothed_classifier.certify(inputs, i, args.N0, args.N, args.alpha, args.batch, dim,
                                                             1000)
        elif args.biased:
            prediction, radius = smoothed_classifier.certify(inputs, i, args.N0, args.N, args.alpha, args.batch, 0,
                                                             1000)
        else:
            prediction, radius = smoothed_classifier.certify(inputs, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()
