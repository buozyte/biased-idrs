How to run the experiments on the cluster and log everything on wandb.

1. Install SEML and read the documentation.
2. Fill in the TODOs in *slurm/idrs_config_TEMPLATE.yaml* and save it as *slurm/idrs_config.yaml* (later we can have as many config files as we want).
3. Log in to the file server, activate the conda environment and navigate to the project directory.
4. Add the jobs to the queue: `seml idrs add slurm/idrs_config.yaml`
5. Start the jobs: `seml idrs start`
6. You can monitor the status of the jobs using
   1. `seml idrs status`
   2. `squeue -u <your_username_on_the_cluster>`
   3. the project page on wandb.ai, 
   4. the output files written into the directory you specified in *idrs_config.yaml*.
   