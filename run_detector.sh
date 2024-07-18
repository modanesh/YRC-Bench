#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --mem=40gb
#SBATCH --gres=gpu:1
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=preprocess
#SBATCH --output=experiments/slurm/%j.out
#SBATCH --time=72:00:00
#SBATCH --qos scavenger
#SBATCH --partition scavenger

eval "$(/nas/ucb/tutrinh/anaconda3/bin/conda shell.bash hook)"
conda activate ood

wandb login 4a6017fc91542ffdb82ee3d6213e9cf0c11fd892

export CUDA_LAUNCH_BLOCKING=1
export CLIPORT_ROOT="/nas/ucb/tutrinh/yield_request_control/cliport/"
cd /nas/ucb/tutrinh/yield_request_control
export PYTHONPATH="$PYTHONPATH:$PWD"

python3 detector_main.py \
	--train \
	--pretrain \
	--ae_model_file /nas/ucb/tutrinh/yield_request_control/logs/train_detector/coinrun/2024-07-18__01-49-42__seed_8888/autoencoder.tar \
	--env_name coinrun \
	--data_dir /nas/ucb/tutrinh/yield_request_control/logs/preprocess_detector/coinrun/2024-07-17__20-30-39/ \
	--device cuda \
	--gpu 3 \
	--seed 8888 \
	--use_wandb
