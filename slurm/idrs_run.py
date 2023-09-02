import time
import os
import sys

import logging
from sacred import Experiment
import seml
import wandb
from wandb.sacred import WandbObserver

# ADD YOUR PACKAGES
import torch
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from train import main_train
from certify import main_certify


project_name = "idrs"
ex = Experiment(project_name)
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))
        ex.observers.append(WandbObserver(config={"project": project_name}, reinit=False))


@ex.automain
def run(dataset_name: str, base_sigma: float, sigma: str, mu: str, _run):
    """ This code is executed for each parameter configuration.
    
    """
    config_dict = locals()
    # if necessary, host name is
    hostname = os.environ.get("HOSTNAME")

    # summary of the configs
    configs_to_print = ("dataset_name", "sigma", "mu", "base_sigma")
    logging.info("Received the following configuration")
    for key in configs_to_print:
        logging.info(f"{key}: {config_dict[key]}")
    experiment_name = "-".join(
        (
            "experiment", dataset_name, str(sigma), str(base_sigma), str(mu)
        )
    )

    collected_results = []

    def wandb_logger(current_results):
        nonlocal collected_results
        wandb.log(current_results)
        collected_results.append(current_results)

    alt_sigmas = {
        0.0: 0.0,
        0.12: 0.126,
        0.25: 0.263,
        0.5: 0.53,
        1.0: 1.0
    }

    if mu is not None:
        mu, mu_weight = mu.split("-", 1)
        mu_weight = float(mu_weight)
    else:
        mu_weight = 0

    out_dir = f"results/{dataset_name}/base_sigma_{base_sigma}"
    if sigma is not None:
        out_dir = os.path.join(out_dir, f'{sigma}')
    if mu is not None:
        out_dir = os.path.join(out_dir, f'{mu}')

    train_params = {
        "out_dir": out_dir,
        "base_sigma": base_sigma,
        "biased": True,
        "bias_weight": 0.0,
        "var_func": sigma,
        "bias_func": mu,
    }
    certify_params = {
        "trained_classifier": os.path.join(out_dir, 'checkpoint.pth.tar'),
        "out_dir": out_dir,
        "base_sigma": base_sigma,
        "biased": True,
        "var_func": sigma,
        "bias_func": mu,
        "bias_weight": mu_weight,
        "external_logger": wandb_logger,
    }
    if dataset_name == "toy":
        train_params["dataset"] = "toy_dataset_linear_sep"
        train_params["arch"] = "linear_model"
        train_params["epochs"] = 10
        train_params["batch"] = 200

        certify_params["dataset"] = "toy_dataset_linear_sep"
        certify_params["index_max"] = 90
        certify_params["batch"] = 200
    elif dataset_name == "cifar10":
        train_params["dataset"] = dataset_name
        train_params["arch"] = "cifar_resnet110"
        train_params["batch"] = 400
        if sigma is not None:
            train_params["alt_sigma_aug"] = alt_sigmas[base_sigma]

        certify_params["dataset"] = dataset_name
        certify_params["batch"] = 400

    logging.info("Start training procedure")
    start = time.time()
    main_train(**train_params)
    end = time.time()
    logging.info(f"Successfully finished training after {end-start} seconds")

    if mu is not None:
        out_dir = os.path.join(out_dir, f'mu_strength_{mu_weight}')

    logging.info("Start certification procedure")
    start = time.time()
    main_certify(**certify_params)
    end = time.time()
    logging.info(f"Successfully finished certification after {end - start} seconds")

    return collected_results
