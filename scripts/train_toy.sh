#!/bin/bash
set -e

# reminder: sbash needs shell config -> for conda
. $HOME/.bashrc

conda activate idrs
# cd $HOME/git/idrs


python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.00/bias_0.0 --num_workers 2 --batch 250 --base_sigma 0.00 --epochs 5 --biased True --bias_func mu_toy --bias_weight 0
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.12/bias_0.0 --num_workers 2 --batch 250 --base_sigma 0.12 --epochs 5 --biased True --bias_func mu_toy --bias_weight 0
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.25/bias_0.0 --num_workers 2 --batch 250 --base_sigma 0.25 --epochs 5 --biased True --bias_func mu_toy --bias_weight 0
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.50/bias_0.0 --num_workers 2 --batch 250 --base_sigma 0.50 --epochs 5 --biased True --bias_func mu_toy --bias_weight 0
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_1.00/bias_0.0 --num_workers 2 --batch 250 --base_sigma 1.00 --epochs 5 --biased True --bias_func mu_toy --bias_weight 0

python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.00/bias_0.3 --num_workers 2 --batch 250 --base_sigma 0.00 --epochs 5 --biased True --bias_func mu_toy --bias_weight 0.3
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.12/bias_0.3 --num_workers 2 --batch 250 --base_sigma 0.12 --epochs 5 --biased True --bias_func mu_toy --bias_weight 0.3
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.25/bias_0.3 --num_workers 2 --batch 250 --base_sigma 0.25 --epochs 5 --biased True --bias_func mu_toy --bias_weight 0.3
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.50/bias_0.3 --num_workers 2 --batch 250 --base_sigma 0.50 --epochs 5 --biased True --bias_func mu_toy --bias_weight 0.3
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_1.00/bias_0.3 --num_workers 2 --batch 250 --base_sigma 1.00 --epochs 5 --biased True --bias_func mu_toy --bias_weight 0.3

python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.00/bias_0.5 --num_workers 2 --batch 250 --base_sigma 0.00 --epochs 5 --biased True --bias_func mu_toy --bias_weight 0.5
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.12/bias_0.5 --num_workers 2 --batch 250 --base_sigma 0.12 --epochs 5 --biased True --bias_func mu_toy --bias_weight 0.5
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.25/bias_0.5 --num_workers 2 --batch 250 --base_sigma 0.25 --epochs 5 --biased True --bias_func mu_toy --bias_weight 0.5
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.50/bias_0.5 --num_workers 2 --batch 250 --base_sigma 0.50 --epochs 5 --biased True --bias_func mu_toy --bias_weight 0.5
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_1.00/bias_0.5 --num_workers 2 --batch 250 --base_sigma 1.00 --epochs 5 --biased True --bias_func mu_toy --bias_weight 0.5
