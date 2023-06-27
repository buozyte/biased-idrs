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


python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.00/bias_0.0 --num_workers 2 --batch 250 --base_sigma 0.00 --epochs 20 --biased True --bias_weight 0
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.12/bias_0.0 --num_workers 2 --batch 250 --base_sigma 0.12 --epochs 20 --biased True --bias_weight 0
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.25/bias_0.0 --num_workers 2 --batch 250 --base_sigma 0.25 --epochs 20 --biased True --bias_weight 0
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.50/bias_0.0 --num_workers 2 --batch 250 --base_sigma 0.50 --epochs 20 --biased True --bias_weight 0
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_1.00/bias_0.0 --num_workers 2 --batch 250 --base_sigma 1.00 --epochs 20 --biased True --bias_weight 0

python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.00/bias_0.3 --num_workers 2 --batch 250 --base_sigma 0.00 --epochs 20 --biased True --bias_weight 0.3
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.12/bias_0.3 --num_workers 2 --batch 250 --base_sigma 0.12 --epochs 20 --biased True --bias_weight 0.3
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.25/bias_0.3 --num_workers 2 --batch 250 --base_sigma 0.25 --epochs 20 --biased True --bias_weight 0.3
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.50/bias_0.3 --num_workers 2 --batch 250 --base_sigma 0.50 --epochs 20 --biased True --bias_weight 0.3
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_1.00/bias_0.3 --num_workers 2 --batch 250 --base_sigma 1.00 --epochs 20 --biased True --bias_weight 0.3

python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.00/bias_0.5 --num_workers 2 --batch 250 --base_sigma 0.00 --epochs 20 --biased True --bias_weight 0.5
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.12/bias_0.5 --num_workers 2 --batch 250 --base_sigma 0.12 --epochs 20 --biased True --bias_weight 0.5
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.25/bias_0.5 --num_workers 2 --batch 250 --base_sigma 0.25 --epochs 20 --biased True --bias_weight 0.5
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.50/bias_0.5 --num_workers 2 --batch 250 --base_sigma 0.50 --epochs 20 --biased True --bias_weight 0.5
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_1.00/bias_0.5 --num_workers 2 --batch 250 --base_sigma 1.00 --epochs 20 --biased True --bias_weight 0.5

python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.00 --num_workers 2 --batch 250 --base_sigma 0.00 --epochs 20
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.12 --num_workers 2 --batch 250 --base_sigma 0.12 --epochs 20
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.25 --num_workers 2 --batch 250 --base_sigma 0.25 --epochs 20
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_0.50 --num_workers 2 --batch 250 --base_sigma 0.50 --epochs 20
python train.py toy_dataset_linear_sep linear_model trained_models/toy/noise_1.00 --num_workers 2 --batch 250 --base_sigma 1.00 --epochs 20
