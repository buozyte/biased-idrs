import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='Plot')
parser.add_argument('path', type=str)
parser.add_argument('filename', type=str)
args = parser.parse_args()


# TODO: adjust path for loading data -> I think it changed a bit
def main():
    """
    Plot the certified accuracy against the radius.

    (Certified accuracy: fraction of data correctly classified under (smoothed) classifier, while prediction certifiably
    robust within ball with radius r)
    """

    # load data
    data = pd.read_csv(f"{args.path}/{args.filename}", delimiter='\t')
    data_id = pd.read_csv(f"{args.path}/{args.filename}_id", delimiter='\t')

    # prepare linspace for radii (x-axis) and storage for certified accuracies (y-axis)
    max_radius = max(data.radius.max(), data_id.radius.max())
    radii = np.linspace(0, max_radius + 0.1, num=100)
    fraction_per_radius = np.ndarray(100)
    fraction_per_radius_id = np.ndarray(100)

    # compute the certified accuracy (w.r.t. each radius in the linspace)
    for i, radius in enumerate(radii):
        robust_set = data[pd.to_numeric(data['radius']) >= radius]
        if len(robust_set) == 0:
            fraction_per_radius[i] = 0
        else:
            fraction_correct = len(robust_set[robust_set.correct == 1])
            fraction_per_radius[i] = fraction_correct / len(data)

        robust_set_id = data_id[pd.to_numeric(data_id['radius']) >= radius]
        if len(robust_set_id) == 0:
            fraction_per_radius_id[i] = 0
        else:
            fraction_correct_id = len(robust_set_id[robust_set_id.correct == 1])
            fraction_per_radius_id[i] = fraction_correct_id / len(data_id)

    # plot and save figure
    plt.plot(radii, fraction_per_radius, '-', label="Cohen")
    plt.plot(radii, fraction_per_radius_id, '-', label="IDRS")
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig(f"{args.path}/{args.filename}_certified_accuracy.pdf")


if __name__ == "__main__":
    main()
