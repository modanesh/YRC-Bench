#!/bin/bash
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

mode=$1
env_name=$2
data_dir=$3
exp_name=$4
gpu_device=$5

if [ "$mode" == "preprocess" ]; then
	python3 detector_main.py \
		--preprocess \
		--format png \
		--env_name ${env_name} \
		--data_dir ${data_dir} \
		--exp_name ${exp_name} \
		--device cuda \
		--gpu ${gpu_device}
elif [ "$mode" == "train" ]; then
	python3 detector_main.py \
		--train \
		--env_name ${env_name} \
		--data_dir ${data_dir} \
		--exp_name ${exp_name} \
		--pretrain \
		--device cuda \
		--gpu ${gpu_device} \
		--seed 8888 \
		--use_wandb
else
    echo "Invalid mode: ${mode}"
    echo "Usage: $0 {preprocess|train} env_name data_dir exp_name gpu_device"
    exit 1
fi
