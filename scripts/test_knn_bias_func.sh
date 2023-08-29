#!/bin/bash
set -e

# reminder: sbash needs shell config -> for conda
. $HOME/.bashrc

conda activate idrs
# cd $HOME/git/idrs

python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.25/bias_0.0 --num_workers 2 --batch 250 --base_sigma 0.25 --epochs 10 --biased True --bias_func mu_knn_based --bias_weight 0

python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.25/bias_0.0/mu_knn_based/checkpoint_biased_id.pth.tar 0.25 data/certify/toy/noise_0.25/bias_0.0 --skip 1 --batch 200 --index_max 90
python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.25/bias_0.0/mu_knn_based/checkpoint_biased_id.pth.tar 0.25 data/certify/toy/noise_0.25/bias_1.0 --skip 1 --batch 200 --biased True --bias_func mu_knn_based --bias_weight 1 --index_max 90

# ---

# python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_0.25 --num_workers 2 --batch 400 --base_sigma 0.25
python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_0.25/bias_0.0 --num_workers 2 --batch 400 --base_sigma 0.25 --biased True --bias_func mu_knn_based --bias_weight 0

python certify.py cifar10 trained_models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar 0.25 data/certify/cifar10/resnet110/noise_0.25/bias_0.0 --skip 20 --batch 400
python certify.py cifar10 trained_models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar 0.25 data/certify/cifar10/resnet110/noise_0.25/bias_1.0 --skip 20 --batch 400 --biased True --bias_func mu_knn_based --bias_weight 1