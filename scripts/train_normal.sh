#!/bin/bash
set -e

# reminder: sbash needs shell config -> for conda
. $HOME/.bashrc

conda activate idrs
# cd $HOME/git/idrs

python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_0.00 --num_workers 2 --batch 400 --base_sigma 0.00
python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_0.12 --num_workers 2 --batch 400 --base_sigma 0.12
python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_0.25 --num_workers 2 --batch 400 --base_sigma 0.25
python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_0.50 --num_workers 2 --batch 400 --base_sigma 0.50
python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_1.00 --num_workers 2 --batch 400 --base_sigma 1.00
