import logging
from sacred import Experiment
import seml
import wandb
from wandb.sacred import WandbObserver

# ADD YOUR PACKAGES
import torch
import numpy as np

import time
import os


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



# ADD FUNCTIONS

def print_random():
    logging.info(f"random number: {np.random.rand()}")
    return 

@ex.automain
def run(model_name: str, dataset_name: str, mu: str, sigma: str, _run):
    """ This code is executed for each parameter configuration.
    
    """
    config_dict = locals()
    device = torch.device("cuda")

    # if necessary, host name is
    hostname = os.environ.get("HOSTNAME")

    # get the model
    #network = get_network(model_name).to(device)

    # summary of the configs
    configs_to_print = ("model_name", "dataset_name")
    logging.info("Received the following configuration")
    for key in configs_to_print:
        logging.info(f"{key}: {config_dict[key]}")
    experiment_name = "-".join(
        (
            "experiment", model_name, dataset_name, mu, sigma
        )
    )

    # dataset
    dataset = [(None, None), (None, None)] # remove
    #dataset = get_dataset(dataset_name)

    # smoothing and verification
    results_list = []
    for i_sample, (X, y) in enumerate(dataset):
        logging.info(f"sample {i_sample}")
        time_start = time.time()

        # check the prediction -> smooth -> verify

        time_overall = time.time()-time_start

        # these results go to wandb (the configs are submitted authomatically)
        results = {
            "sample_index": i_sample,
            "verified_radius": None,
            "lipschitz_constant": None,
            "time": {
                "overall": time_overall
            },
        }
        
        results_list.append(results)
        wandb.log(results)

    # the returned result will be written into the database
    return results_list
