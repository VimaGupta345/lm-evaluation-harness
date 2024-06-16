#!/bin/bash
#SBATCH -J LM_eval_hellaswag                                  # Job name
#SBATCH -A gts-ag117-prism
#SBATCH -q embers
#SBATCH -N1 --gres=gpu:H100:2                               # Number of nodes and GPUs
#SBATCH --mem-per-gpu=80G                                   # Memory per GPU
#SBATCH -t8:00:00                                           # Max time (8 hours)
#SBATCH -o /storage/home/hcoda1/4/vgupta345/p-apadmanabh3-0/lm-eval/lm_eval_new/lm-evaluation-harness/logs/hellaswag/20240424_034441_none-dynamic-drop-2.json/hellaswag_%j.out                        # Output and error file
#SBATCH --mail-type=BEGIN,END,FAIL                          # Mail events
#SBATCH --mail-user=vgupta345@gatech.edu                    # Email for notifications

module load gcc/12.3.0
nvidia-smi                                                                        # validate correct config

source /storage/home/hcoda1/4/vgupta345/micromamba/etc/profile.d/micromamba.sh
micromamba activate /storage/coda1/p-apadmanabh3/0/vgupta345/prowl_new

export NCCL_P2P_DISABLE=1
export TOKENIZERS_PARALLELISM=false

cd /storage/home/hcoda1/4/vgupta345/p-apadmanabh3-0/lm-eval/lm_eval_new/lm-evaluation-harness

lm-eval --model vllm     --model_args pretrained="mistralai/Mixtral-8x7B-v0.1",tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.7,enforce_eager=True,mixtral_config_file="/storage/coda1/p-apadmanabh3/0/vgupta345/prowl/mixtral_configs/none-dynamic-drop-2.json"     --tasks hellaswag     --num_fewshot 10     --batch_size 128     --wandb_args project=prowl,group=acc_drop_2,name=hellaswag     >> "/storage/home/hcoda1/4/vgupta345/p-apadmanabh3-0/lm-eval/lm_eval_new/lm-evaluation-harness/logs/hellaswag/20240424_034441_none-dynamic-drop-2.json/lm_eval_hellaswag.log"

micromamba deactivate
