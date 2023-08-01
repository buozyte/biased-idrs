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

python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.00/bias_0.0/checkpoint_biased.pth.tar 0.00 data/certify/toy/noise_0.00/train_bias_0.0 --skip 1 --batch 200 --biased True --bias_weight 0.0
python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.00/bias_0.3/checkpoint_biased.pth.tar 0.00 data/certify/toy/noise_0.00/train_bias_0.3 --skip 1 --batch 200 --biased True --bias_weight 0.3
python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.00/bias_0.5/checkpoint_biased.pth.tar 0.00 data/certify/toy/noise_0.00/train_bias_0.5 --skip 1 --batch 200 --biased True --bias_weight 0.5

python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.12/bias_0.0/checkpoint_biased.pth.tar 0.12 data/certify/toy/noise_0.12/train_bias_0.0 --skip 1 --batch 200 --biased True --bias_weight 0.0
python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.12/bias_0.3/checkpoint_biased.pth.tar 0.12 data/certify/toy/noise_0.12/train_bias_0.3 --skip 1 --batch 200 --biased True --bias_weight 0.3
python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.12/bias_0.5/checkpoint_biased.pth.tar 0.12 data/certify/toy/noise_0.12/train_bias_0.5 --skip 1 --batch 200 --biased True --bias_weight 0.5

python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.25/bias_0.0/checkpoint_biased.pth.tar 0.25 data/certify/toy/noise_0.25/train_bias_0.0 --skip 1 --batch 200 --biased True --bias_weight 0.0
python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.25/bias_0.3/checkpoint_biased.pth.tar 0.25 data/certify/toy/noise_0.25/train_bias_0.3 --skip 1 --batch 200 --biased True --bias_weight 0.3
python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.25/bias_0.5/checkpoint_biased.pth.tar 0.25 data/certify/toy/noise_0.25/train_bias_0.5 --skip 1 --batch 200 --biased True --bias_weight 0.5

python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.50/bias_0.0/checkpoint_biased.pth.tar 0.50 data/certify/toy/noise_0.50/train_bias_0.0 --skip 1 --batch 200 --biased True --bias_weight 0.0
python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.50/bias_0.3/checkpoint_biased.pth.tar 0.50 data/certify/toy/noise_0.50/train_bias_0.3 --skip 1 --batch 200 --biased True --bias_weight 0.3
python certify.py toy_dataset_linear_sep trained_models/toy/noise_0.50/bias_0.5/checkpoint_biased.pth.tar 0.50 data/certify/toy/noise_0.50/train_bias_0.5 --skip 1 --batch 200 --biased True --bias_weight 0.5
