#!/bin/bash
set -e

# reminder: sbash needs shell config -> for conda

conda activate idrs
cd $HOME/git/idrs

python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_0.12 --num_workers 2 --batch 400 --base_sigma 0.12 --id_var True --alt_sigma_aug 0.126
python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_0.25 --num_workers 2 --batch 400 --base_sigma 0.25 --id_var True --alt_sigma_aug 0.263
python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_0.50 --num_workers 2 --batch 400 --base_sigma 0.50 --id_var True --alt_sigma_aug 0.53
