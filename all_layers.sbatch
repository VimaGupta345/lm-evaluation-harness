#!/bin/bash
#SBATCH -J LM_eval_all_layers                                # Job name
#SBATCH -A gts-ag117
#SBATCH -q embers
#SBATCH -N1 --gres=gpu:A100:1                               # Number of nodes, GPUs, and cores required
#SBATCH --mem-per-gpu=80G                                   # Memory per gpu
#SBATCH -t6:00:00                                        # Duration of the job (Ex: 15 mins)
#SBATCH -o all_layers_%j.out                             # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL                          # Mail preferences
#SBATCH --mail-user=vgupta345@gatech.edu                     # e-mail address for notifications

nvidia-smi                                                                        # validate correct config

source /storage/home/hcoda1/4/vgupta345/micromamba/etc/profile.d/micromamba.sh
micromamba activate /storage/coda1/p-apadmanabh3/0/vgupta345/prowl_new

cd /storage/home/hcoda1/4/vgupta345/p-apadmanabh3-0/lm-eval/lm_eval_new/lm-evaluation-harness/

export TASKS="arc-challenge,hellaswag,truthfulqa-mc,mmlu,winogrande,gsm8k"

# You can now use $TASKS in your commands to refer to this list of tasks.
# Below are example commands for each task with hypothetical few-shot settings.

lm-eval --model vllm \
    --model_args pretrained="mistralai/Mixtral-8x7B-v0.1",tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.7,enforce_eager=True,mixtral_config_file="$MIXTRAL_CONFIG_PATH" \
    --tasks arc-challenge \
    --num_fewshot 25 \
    --batch_size auto \
    --wandb_args project=prowl,group=arc-challenge \
    >> "lm_eval_arc-challenge.log"

lm-eval --model vllm \
    --model_args pretrained="mistralai/Mixtral-8x7B-v0.1",tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.7,enforce_eager=True,mixtral_config_file="$MIXTRAL_CONFIG_PATH" \
    --tasks hellaswag \
    --num_fewshot 10 \
    --batch_size auto \
    --wandb_args project=prowl,group=hellaswag \
    >> "lm_eval_hellaswag.log"

lm-eval --model vllm \
    --model_args pretrained="mistralai/Mixtral-8x7B-v0.1",tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.7,enforce_eager=True,mixtral_config_file="$MIXTRAL_CONFIG_PATH" \
    --tasks truthfulqa-mc \
    --num_fewshot 0 \
    --batch_size auto \
    --wandb_args project=prowl,group=truthfulqa-mc \
    >> "lm_eval_truthfulqa-mc.log"

lm-eval --model vllm \
    --model_args pretrained="mistralai/Mixtral-8x7B-v0.1",tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.7,enforce_eager=True,mixtral_config_file="$MIXTRAL_CONFIG_PATH" \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size auto \
    --wandb_args project=prowl,group=mmlu \
    >> "lm_eval_mmlu.log"

lm-eval --model vllm \
    --model_args pretrained="mistralai/Mixtral-8x7B-v0.1",tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.7,enforce_eager=True,mixtral_config_file="$MIXTRAL_CONFIG_PATH" \
    --tasks winogrande \
    --num_fewshot 5 \
    --batch_size auto \
    --wandb_args project=prowl,group=winogrande \
    >> "lm_eval_winogrande.log"

lm-eval --model vllm \
    --model_args pretrained="mistralai/Mixtral-8x7B-v0.1",tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.7,enforce_eager=True,mixtral_config_file="$MIXTRAL_CONFIG_PATH" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size auto \
    --wandb_args project=prowl,group=gsm8k \
    >> "lm_eval_gsm8k.log"


micromamba deactivate




