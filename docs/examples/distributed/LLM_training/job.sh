#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=a100:2
#SBATCH --mem=512G
#SBATCH --time=01:00:00
#SBATCH --job-name=llm_training

set -e  # exit on error.

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Ensure only anaconda/3 module loaded.
module --quiet purge
# This example uses Conda to manage package dependencies.
# See https://docs.mila.quebec/Userguide.html#conda for more information.
module load anaconda/3
module load cuda/11.7


# CONDA_ENV_PREFIX=$SLURM_TMPDIR/env
CONDA_ENV_PREFIX=$SCRATCH/conda/llm_training


if [ ! -d $CONDA_ENV_PREFIX ]; then
    # Create a conda environment and use the libmamba solver:
    conda create -y -p $CONDA_ENV_PREFIX python=3.9 conda conda-libmamba-solver -c conda-forge
    conda activate $CONDA_ENV_PREFIX
    export CONDA_EXE="$(hash -r; which conda)"
    conda config --set solver libmamba

    # Install pytorch:
    conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    # Install other conda packages:
    # conda install -y rich -c conda-forge
    # Install other pip packages:
    pip install rich transformers datasets evaluate accelerate deepspeed
else
    conda activate $CONDA_ENV_PREFIX
fi

output_dir=$SCRATCH/logs/llm_training/$SLURM_JOB_ID
mkdir -p $output_dir
conda env export > $output_dir/environment.yml

set -x  # print commands.

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_ON_NODE))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

gpus_per_task=$(($WORLD_SIZE / $SLURM_NTASKS))
cpus_per_gpu=$(($SLURM_CPUS_PER_TASK / $gpus_per_task))

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:="configs/ds_level2.yaml"}

# memory_available=
mem_limit_in_bytes=$(cat /sys/fs/cgroup/memory/slurm/uid_"$(id -u)"/job_"${SLURM_JOBID}"/memory.limit_in_bytes)
# TODO: Load the dataset in-memory:
export HF_DATASETS_IN_MEMORY_MAX_SIZE=$mem_limit_in_bytes


# Run `accelerate launch (...)` on each node:
# cmd=(accelerate launch
#         --config_file="$ACCELERATE_CONFIG"
#         --machine_rank="$SLURM_NODEID"
#         --num_cpu_threads_per_process="$cpus_per_gpu"
#         --main_process_ip="$MASTER_ADDR"
#         --main_process_port="$MASTER_PORT"
#         --num_processes="$WORLD_SIZE"
#         main.py
#         --output_dir="$output_dir"
#         "$@")

# srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
#     bash -c "$(for a in "${cmd[@]}" ; do echo -n \"$a\" "" ; done)"

# Run `accelerate launch (...)` on each node:
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c 'accelerate launch \
    --config_file='$ACCELERATE_CONFIG' \
    --machine_rank=$SLURM_NODEID \
    --num_cpu_threads_per_process='$SLURM_CPUS_PER_TASK' \
    --main_process_ip='$MASTER_ADDR' \
    --main_process_port='$MASTER_PORT' \
    --num_processes='$WORLD_SIZE' \
    main.py \
    --output_dir='$output_dir' \
    --config_name=facebook/opt-2.7b --tokenizer_name=facebook/opt-2.7b \
    --dataset_name=wikitext --dataset_config_name wikitext-103-v1 \
    --per_device_train_batch_size=1 --max_train_steps=1000 --with_tracking --report_to=wandb'
