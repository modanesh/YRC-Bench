#!/bin/bash

env_name=$1
help_option=$2
query_costs=("5.0")

for query_cost in "${query_costs[@]}"; do
    cmd="CUDA_VISIBLE_DEVICES=6 ./eval_skyline.sh $env_name $help_option $query_cost"
    echo "Running: $cmd"
    eval $cmd
done
