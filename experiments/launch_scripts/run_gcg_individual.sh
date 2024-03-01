#!/bin/bash

#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export model=$1 # llama2 or vicuna
export setup=$2 # behaviors or strings

# Create results folder if it doesn't exist
if [ ! -d "../results_spe" ]; then
    mkdir "../results_spe"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

for data_offset in 0
do

    python -u ../main.py \
        --config="../configs/individual_${model}.py" \
        --config.attack=gcg \
        --config.train_data="../../data/advbench/harmful_${setup}.csv" \
        --config.result_prefix="../results/individual_abla_mistral_${setup}_${model}_gcg_offset${data_offset}" \
        --config.n_train_data=100 \
        --config.data_offset=$data_offset \
        --config.n_steps=1000 \
        --config.test_steps=50 \
        --config.batch_size=512 \
        --config.probe_set=64 \
        --config.filtered_set=32

done