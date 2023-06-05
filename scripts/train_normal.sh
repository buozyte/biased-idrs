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

python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_0.00 --num_workers 2 --batch 400 --base_sigma 0.00
python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_0.12 --num_workers 2 --batch 400 --base_sigma 0.12
python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_0.25 --num_workers 2 --batch 400 --base_sigma 0.25
python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_0.50 --num_workers 2 --batch 400 --base_sigma 0.50
python train.py cifar10 cifar_resnet110 trained_models/cifar10/resnet110/noise_1.00 --num_workers 2 --batch 400 --base_sigma 1.00
