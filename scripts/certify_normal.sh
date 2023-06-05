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

python certify.py cifar10 trained_models/cifar10/resnet110/noise_0.12/checkpoint.pth.tar 0.12 data/certify/cifar10/resnet110/noise_0.12/test --skip 20 --batch 400
python certify.py cifar10 trained_models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar 0.25 data/certify/cifar10/resnet110/noise_0.25/test --skip 20 --batch 400
python certify.py cifar10 trained_models/cifar10/resnet110/noise_0.50/checkpoint.pth.tar 0.50 data/certify/cifar10/resnet110/noise_0.50/test --skip 20 --batch 400
python certify.py cifar10 trained_models/cifar10/resnet110/noise_1.00/checkpoint.pth.tar 1.00 data/certify/cifar10/resnet110/noise_1.00/test --skip 20 --batch 400
