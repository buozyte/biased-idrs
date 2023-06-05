#!/bin/bash
set -e

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/nfs/homedirs/buozyte/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/nfs/homedirs/buozyte/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/nfs/homedirs/buozyte/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/nfs/homedirs/buozyte/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate idrs
cd $HOME/git/idrs

echo "Start training"
python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_0.00 --num_workers 2 --batch 400 --base_sigma 0.00
python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_0.12 --num_workers 2 --batch 400 --base_sigma 0.12
python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_0.25 --num_workers 2 --batch 400 --base_sigma 0.25
python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_0.50 --num_workers 2 --batch 400 --base_sigma 0.50
python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_1.00 --num_workers 2 --batch 400 --base_sigma 1.00

python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_0.12 --num_workers 2 --batch 400 --base_sigma 0.12 --input_dependent True --alt_sigma_aug 0.126
python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_0.25 --num_workers 2 --batch 400 --base_sigma 0.25 --input_dependent True --alt_sigma_aug 0.263
python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_0.50 --num_workers 2 --batch 400 --base_sigma 0.50 --input_dependent True --alt_sigma_aug 0.53

echo "Start certifying"
python certify.py cifar10 trained_models/cifar10/resnet110/noise_0.12/checkpoint.pth.tar 0.12 data/certify/cifar10/resnet110/noise_0.12/test --skip 20 --batch 400
python certify.py cifar10 trained_models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar 0.25 data/certify/cifar10/resnet110/noise_0.25/test --skip 20 --batch 400
python certify.py cifar10 trained_models/cifar10/resnet110/noise_0.50/checkpoint.pth.tar 0.50 data/certify/cifar10/resnet110/noise_0.50/test --skip 20 --batch 400
python certify.py cifar10 trained_models/cifar10/resnet110/noise_1.00/checkpoint.pth.tar 1.00 data/certify/cifar10/resnet110/noise_1.00/test --skip 20 --batch 400

python certify.py cifar10 trained_models/cifar10/resnet110/noise_0.12/checkpoint_id.pth.tar 0.12 data/certify/cifar10/resnet110/noise_0.12/test --skip 20 --batch 400 --input_dependent True
python certify.py cifar10 trained_models/cifar10/resnet110/noise_0.25/checkpoint_id.pth.tar 0.25 data/certify/cifar10/resnet110/noise_0.25/test --skip 20 --batch 400 --input_dependent True
python certify.py cifar10 trained_models/cifar10/resnet110/noise_0.50/checkpoint_id.pth.tar 0.50 data/certify/cifar10/resnet110/noise_0.50/test --skip 20 --batch 400 --input_dependent True

echo "Start certifying with mismatching sigmas"
python certify.py cifar10 trained_models/cifar10/resnet110/noise_0.00/checkpoint.pth.tar 0.50 data/certify/cifar10/resnet110/noise_0.00/test --skip 20 --batch 400
python certify.py cifar10 trained_models/cifar10/resnet110/noise_0.12/checkpoint.pth.tar 0.50 data/certify/cifar10/resnet110/noise_0.12/test --skip 20 --batch 400
python certify.py cifar10 trained_models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar 0.50 data/certify/cifar10/resnet110/noise_0.25/test --skip 20 --batch 400
python certify.py cifar10 trained_models/cifar10/resnet110/noise_1.00/checkpoint.pth.tar 0.50 data/certify/cifar10/resnet110/noise_1.00/test --skip 20 --batch 400

python certify.py cifar10 trained_models/cifar10/resnet110/noise_0.12/checkpoint_id.pth.tar 0.50 data/certify/cifar10/resnet110/noise_0.12/test --skip 20 --batch 400 --input_dependent True
python certify.py cifar10 trained_models/cifar10/resnet110/noise_0.25/checkpoint_id.pth.tar 0.50 data/certify/cifar10/resnet110/noise_0.25/test --skip 20 --batch 400 --input_dependent True

echo "Finished script"
