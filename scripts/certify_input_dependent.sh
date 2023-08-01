#!/bin/bash
set -e

# reminder: sbash needs shell config -> for conda

conda activate idrs
cd $HOME/git/idrs

python certify.py cifar10 trained_models/cifar10/resnet110/noise_0.12/checkpoint_id.pth.tar 0.12 data/certify/cifar10/resnet110/noise_0.12 --skip 20 --batch 400 --id_var True
python certify.py cifar10 trained_models/cifar10/resnet110/noise_0.25/checkpoint_id.pth.tar 0.25 data/certify/cifar10/resnet110/noise_0.25 --skip 20 --batch 400 --id_var True
python certify.py cifar10 trained_models/cifar10/resnet110/noise_0.50/checkpoint_id.pth.tar 0.50 data/certify/cifar10/resnet110/noise_0.50 --skip 20 --batch 400 --id_var True
