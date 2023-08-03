# Biased Input-Dependent Randomized Smoothing

## Overview

The code provided in this repository can be used to train and certify a smoothed classifier based on the concept of
randomized smoothing. It combines and extends the previously presented implementations for the normal randomized
smoothing approach by J. Cohen et al. (["Certified Adversarial Robustness via Randomized
Smoothing"](http://proceedings.mlr.press/v97/cohen19c/cohen19c.pdf)), and the modified randomized smoothing approach
including an input-dependent variance by P. Sukenik et al. (["Intriguing Properties of Input-Dependent Randomized
Smoothing"](https://arxiv.org/pdf/2110.05365.pdf)). The [randomized
smoothing](https://gitlab.lrz.de/ga27fey/idrs/-/blob/main/models/rs.py) approach and [input-dependent randomized
smoothing](https://gitlab.lrz.de/ga27fey/idrs/-/blob/main/models/input_dependent_rs.py) approach can be found in the
`models` directory.

The extension in based on the input-dependent approach and includes a non-zero, input-dependent bias.
[This approach](https://gitlab.lrz.de/ga27fey/idrs/-/blob/main/models/biased_idrs.py) can be found in the `models`
directory, as well.

## Structure of Repository

The `models` directory includes all three version of the smoothing approaches, as well as a collection of base models,
which will be trained and then smoothed according to the chosen approach. Different functions, that can be used to
compute the input-dependent bias (`mu`) or variance (`sigma`), can be found in the according files in the 
`input_dependent_functions` directory.

The files provided in the `scripts` directory can be used to run a training, certification or prediction. Alternatively,
the three scripts `train.py`, `certify.py` and `predict.py` can be used to run those jobs.

## Prerequisites

### Set up `venv`

Create a suitable `venv` (for example using `conda`) with the Python version 3.7, 3.8 or 3.9 and install the requirements
provided in the repository into this environment (`pip install -r requirements.txt`).

Note that the installation of `R` into the environment is needed to install and use some packages.

## Usage

To train models and certify or test smoothed models use the accordingly named scripts. Further, notes on the training
and certification can be found below.

Notes:
- In general the implementation for the biased IDRS approach can be used instead of the implementation of the approaches
  by either not providing any functions or only a function for the variance.
- By setting the `biased` parameter to `True`, the biased IDRS scheme is used with the chosen function for the bias and
  variance. If no bias function is provided, the bias is set to 0. If no variance function is provided, the variance is
  so to the constant `base sigma`.
- Currently, the dynamic choice of the bias and variance function is only available for the biased IDRS method, not the
  non-biased one, as it can be translated to the biased scheme (by not providing a bias function).

### Adding new functions for the variance and bias

_TODO_: "dynamically" change the function used to compute the variance and bias for the noise used on the training data.

1. Add the function in the according file in the `input_dependent_functions` directory and add its name to the list of
allowed functions, which can be found on the top of each file.
   
2. Add the according new case in either the `bias_id` or `sigma_id` function in the [biased IDRS
   model](https://gitlab.lrz.de/ga27fey/idrs/-/blob/main/models/biased_idrs.py).
   
3. In case new parameters are needed to compute the new bias or variance, add the according parameters to the
   initialization function of the class. Modify all necessary files, especially `train.py`, `certify.py` and
   `predict.py`, to include the new parameters in the definition of an instance of that class.
   
4. Adapt the training according to the new function, if needed.

### Training a model

To train a model, the [`train.py`](https://gitlab.lrz.de/ga27fey/idrs/-/blob/main/train.py) script can be used.
Parameters are provided via argparse. Necessary and optional parameters can be found in either the file or by using the
`--help` method. Examples can be found in the `scripts` directory.

### Certifying a model

Before certifying a model, make sure that the training for the according model went through successfully.

To certify a model, the [`certify.py`](https://gitlab.lrz.de/ga27fey/idrs/-/blob/main/certify.py) script can be used.
Parameters are provided via argparse. Necessary and optional parameters can be found in either the file or by using the
`--help` method. Examples can be found in the `scripts` directory.

### Running the [scripts](https://gitlab.lrz.de/ga27fey/idrs/-/tree/main/scripts) using `slurm`

Tested utilities: `sbatch`, `srun`.

Issue encountered with `sbatch`: if `.bashrc` configured to skip conda initialization in a non-interactive mode,
activating conda venvs in bash scripts will fail. Solve by moving conda initialization or not using conda venvs
(-> adapt scripts accordingly).

**Notes**:
- assumptions: conda venv is named `idrs` and job is started from the top level directory of the repo.
- depending on where the results should be saved, the scripts have to adjusted

_`sbatch` example:_
```
sbatch
-o {path-to-output-file}
-e {path-to-error-file}
-- {further parameters}
{path-to-script}
```
```
sbatch -o $HOME/train.out -e $HOME/train.err --time=0-08:00 --cpus-per-task=2 --mem=16G --gres=gpu:1 --partition=gpu_all --qos=students $HOME/git/idrs/scripts/train_input_dependent.sh
```
