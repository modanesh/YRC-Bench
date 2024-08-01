#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=40gb
#SBATCH --gres=gpu:1
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=heist_skyline
#SBATCH --output=experiments/slurm/%j.out
#SBATCH --time=36:00:00
#SBATCH --qos scavenger
#SBATCH --partition scavenger

eval "$(/nas/ucb/tutrinh/anaconda3/bin/conda shell.bash hook)"
conda activate ood

wandb login 4a6017fc91542ffdb82ee3d6213e9cf0c11fd892

export CUDA_LAUNCH_BLOCKING=1
export CLIPORT_ROOT="/nas/ucb/tutrinh/yield_request_control/cliport/"
cd /nas/ucb/tutrinh/yield_request_control
export PYTHONPATH="$PYTHONPATH:$PWD"

env_name=$1
help_option=$2
query_cost=$3
if [ "$env_name" == "maze" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_aisc/model_200015872.pth"
    expert_model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze/model_200015872.pth"
elif [ "$env_name" == "maze_yellowstar_redgem" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_redline_yellowgem/model_200015872.pth"
    expert_model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_yellowstar_redgem/model_200015872.pth"
elif [ "$env_name" == "heist_aisc_many_keys" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/heist_aisc_many_chests/model_200015872.pth"
    expert_model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/heist_aisc_many_keys/model_200015872.pth"
elif [ "$env_name" == "coinrun_aisc" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/coinrun/model_200015872.pth"
    expert_model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/coinrun_aisc/model_200015872.pth"
fi

python3 test.py \
    --benchmark procgen \
    --env_name ${env_name} \
    --param_name hard_plus \
    --weak_model_file ${model_file} \
    --strong_model_file ${expert_model_file} \
    --help_policy_type ${help_option} \
    --strong_query_cost ${query_cost} \
    --switching_cost 0
