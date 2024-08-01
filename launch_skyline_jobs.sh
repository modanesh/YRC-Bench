#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <environment>"
    exit 1
fi

env_name=$1

combinations=(
    "T1 0"
    "T1 10"
    "T1 20"
    "T1 50"
    "T2 0"
    "T2 10"
    "T2 20"
    "T2 50"
    "T3 0"
    "T3 10"
    "T3 20"
    "T3 50"
)

for combination in "${combinations[@]}"; do
    help_option=$(echo $combination | cut -d ' ' -f 1)
    query_cost=$(echo $combination | cut -d ' ' -f 2)
    sbatch_cmd="sbatch train_skyline.sh $env_name $help_option $query_cost"
    job_id=$($sbatch_cmd | awk '{print $4}')
    echo "Submitted batch job $job_id with parameters: env_name=$env_name, help_option=$help_option, query_cost=$query_cost"
done

