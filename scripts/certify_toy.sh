#!/bin/bash
set -e

# reminder: sbash needs shell config -> for conda
. $HOME/.bashrc

conda activate idrs
# cd $HOME/git/idrs

python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.00/bias_0.0/mu_toy/checkpoint_biased_id.pth.tar 0.00 data/certify/toy/noise_0.00/train_bias_0.0 --skip 1 --batch 200 --biased True --bias_func mu_toy --bias_weight 0.0
python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.00/bias_0.3/mu_toy/checkpoint_biased_id.pth.tar 0.00 data/certify/toy/noise_0.00/train_bias_0.3 --skip 1 --batch 200 --biased True --bias_func mu_toy --bias_weight 0.3
python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.00/bias_0.5/mu_toy/checkpoint_biased_id.pth.tar 0.00 data/certify/toy/noise_0.00/train_bias_0.5 --skip 1 --batch 200 --biased True --bias_func mu_toy --bias_weight 0.5

python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.12/bias_0.0/mu_toy/checkpoint_biased_id.pth.tar 0.12 data/certify/toy/noise_0.12/train_bias_0.0 --skip 1 --batch 200 --biased True --bias_func mu_toy --bias_weight 0.0
python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.12/bias_0.3/mu_toy/checkpoint_biased_id.pth.tar 0.12 data/certify/toy/noise_0.12/train_bias_0.3 --skip 1 --batch 200 --biased True --bias_func mu_toy --bias_weight 0.3
python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.12/bias_0.5/mu_toy/checkpoint_biased_id.pth.tar 0.12 data/certify/toy/noise_0.12/train_bias_0.5 --skip 1 --batch 200 --biased True --bias_func mu_toy --bias_weight 0.5

python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.25/bias_0.0/mu_toy/checkpoint_biased_id.pth.tar 0.25 data/certify/toy/noise_0.25/train_bias_0.0 --skip 1 --batch 200 --biased True --bias_func mu_toy --bias_weight 0.0
python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.25/bias_0.3/mu_toy/checkpoint_biased_id.pth.tar 0.25 data/certify/toy/noise_0.25/train_bias_0.3 --skip 1 --batch 200 --biased True --bias_func mu_toy --bias_weight 0.3
python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.25/bias_0.5/mu_toy/checkpoint_biased_id.pth.tar 0.25 data/certify/toy/noise_0.25/train_bias_0.5 --skip 1 --batch 200 --biased True --bias_func mu_toy --bias_weight 0.5

python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.50/bias_0.0/mu_toy/checkpoint_biased_id.pth.tar 0.50 data/certify/toy/noise_0.50/train_bias_0.0 --skip 1 --batch 200 --biased True --bias_func mu_toy --bias_weight 0.0
python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.50/bias_0.3/mu_toy/checkpoint_biased_id.pth.tar 0.50 data/certify/toy/noise_0.50/train_bias_0.3 --skip 1 --batch 200 --biased True --bias_func mu_toy --bias_weight 0.3
python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.50/bias_0.5/mu_toy/checkpoint_biased_id.pth.tar 0.50 data/certify/toy/noise_0.50/train_bias_0.5 --skip 1 --batch 200 --biased True --bias_func mu_toy --bias_weight 0.5
