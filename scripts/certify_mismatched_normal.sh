#!/bin/bash
set -e

# reminder: sbash needs shell config -> for conda
. $HOME/.bashrc

conda activate idrs
# cd $HOME/git/idrs

python certify.py cifar10 trained_models/cifar10/resnet110/noise_0.00/checkpoint.pth.tar 0.50 data/certify/cifar10/resnet110/noise_0.00/test --skip 20 --batch 400
python certify.py cifar10 trained_models/cifar10/resnet110/noise_0.12/checkpoint.pth.tar 0.50 data/certify/cifar10/resnet110/noise_0.12/test --skip 20 --batch 400
python certify.py cifar10 trained_models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar 0.50 data/certify/cifar10/resnet110/noise_0.25/test --skip 20 --batch 400
python certify.py cifar10 trained_models/cifar10/resnet110/noise_1.00/checkpoint.pth.tar 0.50 data/certify/cifar10/resnet110/noise_1.00/test --skip 20 --batch 400
