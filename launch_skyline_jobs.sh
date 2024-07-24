#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <environment>"
    exit 1
fi

env_name=$1

combinations=(
    "T1 0.1"
    "T1 0.5"
    "T1 1"
    "T1 2"
    "T2 0.1"
    "T2 0.5"
    "T2 1"
    "T2 2"
    "T3 0.1"
    "T3 0.5"
    "T3 1"
    "T3 2"
)

for combination in "${combinations[@]}"; do
    help_option=$(echo $combination | cut -d ' ' -f 1)
    query_cost=$(echo $combination | cut -d ' ' -f 2)
    sbatch_cmd="sbatch train_skyline.sh $env_name $help_option $query_cost"
    job_id=$($sbatch_cmd | awk '{print $4}')
    echo "Submitted batch job $job_id with parameters: env_name=$env_name, help_option=$help_option, query_cost=$query_cost"
done

