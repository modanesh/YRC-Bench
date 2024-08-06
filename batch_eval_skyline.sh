#!/bin/bash

env_name=$1
help_option=$2
query_costs=("0.0" "0.1" "0.5" "1.0" "2.0" "10" "20" "50")

for query_cost in "${query_costs[@]}"; do
    cmd="CUDA_VISIBLE_DEVICES=7 ./eval_skyline.sh $env_name $help_option $query_cost"
    echo "Running: $cmd"
    eval $cmd
done
