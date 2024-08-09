#!/bin/bash

#SBATCH --job-name=vllm_eval_noise
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=5G
#SBATCH --gres=gpu:hgx:2
#SBATCH -p pot
#SBATCH -o ./log/%x-%j.log

# Assign GPUs
# export CUDA_VISIBLE_DEVICES=6,7

# Parameters
NNODES=1
GPUS_PER_NODE=2
# 获取分配给作业的 GPU 数量
gpu_list=$SLURM_JOB_GPUS

# 将逗号分隔的字符串转换为数组
IFS=',' read -ra gpu_array <<< "$gpu_list"

# 获取 GPU 数量
num_gpus=${#gpu_array[@]}

# Conda environment activation (if needed)
# conda activate vllm

# Model paths
declare -A model_dict
model_dict['chatglm3']="Your model path"
# Dataset choices
# datasets=('en' 'zh' 'en_int' 'zh_int' 'en_fact' 'zh_fact')
datasets=('zh')

# Noise rate choices
noise_rates=(0.0 0.2 0.4 0.6 0.8)

# Loop over models
for model_name in "${!model_dict[@]}"
do
    echo "$model_name : ${model_dict[$model_name]}"
    
    for dataset_choice in "${datasets[@]}"
    do
        echo "Running with --dataset $dataset_choice"

        for noise_rate_choice in "${noise_rates[@]}"
        do
            echo "Running with --noise_rate $noise_rate_choice"

            python evalue_prompt.py \
                --gpu_num $num_gpus \
                --model_path "${model_dict[$model_name]}" \
                --type "$model_name" \
                --beam 1 \
                --max_length 8192 \
                --dataset "$dataset_choice" \
                --temp 0.2 \
                --noise_rate "$noise_rate_choice" \
                --passage_num 5 \

            echo "Finished with --noise_rate $noise_rate_choice"
        done

        echo "Finished with --dataset $dataset_choice"
    done

    echo "$model_name done."
done
