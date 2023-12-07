from pathlib import Path

DATASETS = ["cifar10", "toy_dataset_linear_sep", "toy_dataset_blobs", "toy_dataset_cone_shaped"]
PATH_DATA = Path("/nfs/shared/data/")

ALT_SIGMAS = {
    0.0: 0.0,
    0.12: 0.126,
    0.25: 0.263,
    0.5: 0.53,
    1.0: 1.0
}


def PARAMETERS_DEPENDING_ON_DATASET(dataset_name, **kwargs):
    train_params, certify_params = {}, {}

    if dataset_name == "toy":
        train_params["dataset"] = "toy_dataset_linear_sep"
        train_params["arch"] = "linear_model"
        train_params["epochs"] = 10
        train_params["batch"] = 200

        certify_params["dataset"] = "toy_dataset_linear_sep"
        certify_params["index_max"] = 90
        certify_params["batch"] = 200
    if dataset_name == "toy_blobs":
        train_params["dataset"] = "toy_dataset_blobs"
        train_params["arch"] = "linear_blob_model"
        train_params["epochs"] = 5
        train_params["batch"] = 200

        certify_params["dataset"] = "toy_dataset_blobs"
        certify_params["index_max"] = 90
        certify_params["batch"] = 200
    elif dataset_name == "cone_toy":
        train_params["dataset"] = "toy_dataset_cone_shaped"
        train_params["arch"] = "toy_model"
        train_params["epochs"] = 10
        train_params["batch"] = 200

        certify_params["dataset"] = "toy_dataset_cone_shaped"
        certify_params["index_max"] = 90
        certify_params["batch"] = 200
    elif dataset_name == "cifar10":
        train_params["dataset"] = dataset_name
        train_params["arch"] = "cifar_resnet110"
        train_params["batch"] = 400
        if kwargs["sigma"] is not None:
            train_params["alt_sigma_aug"] = ALT_SIGMAS[kwargs["base_sigma"]]

        certify_params["dataset"] = dataset_name
        certify_params["batch"] = 400

    if kwargs["mu"] == "mu_knn_based":
        certify_params["lipschitz_const"] = 3.0 * ["mu_weight"]
    if kwargs["mu"].startswith("mu_constant"):
        certify_params["lipschitz_const"] = 0.0

    return train_params, certify_params
