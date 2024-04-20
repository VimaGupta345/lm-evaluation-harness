#!/bin/bash
#SBATCH -J LM_eval_merge_attn_all_slerp_0.7                                # Job name
#SBATCH -A gts-ag117
#SBATCH -q embers
#SBATCH -N1 --gres=gpu:A100:1                               # Number of nodes, GPUs, and cores required
#SBATCH --mem-per-gpu=80G                                   # Memory per gpu
#SBATCH -t6:00:00                                        # Duration of the job
#SBATCH -o merge_attn_all_slerp_0.7_%j.out                             # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL                          # Mail preferences
#SBATCH --mail-user=vgupta345@gatech.edu                     # e-mail address for notifications

nvidia-smi                                                                        # validate correct config

source /storage/home/hcoda1/4/vgupta345/micromamba/etc/profile.d/micromamba.sh
micromamba activate /storage/coda1/p-apadmanabh3/0/vgupta345/prowl_new

cd /storage/coda1/p-apadmanabh3/0/vgupta345/merging_exp/mergekit

mergekit-yaml merging_experiments/slerp_0.7/slerp_merge_attn_all.yaml ../slerp_0.7_Wizard7B_Vicuna7b/merge_attn_all/

micromamba deactivate

micromamba activate /storage/coda1/p-apadmanabh3/0/vgupta345/new_eval

cd /storage/home/hcoda1/4/vgupta345/p-apadmanabh3-0/lm-eval/lm_eval_new/lm-evaluation-harness/

lm_eval --model vllm  --model_args pretrained=/storage/coda1/p-apadmanabh3/0/vgupta345/merging_exp/slerp_0.7_Wizard7B_Vicuna7b/merge_attn_all --tasks mmlu,lambada_openai,gsm8k,truthfulqa --batch_size auto --wandb_args project=model_merging,name=wv-0.7-merge_attn_all |& tee merge_0.7_merge_attn_all_new_tasks.log

micromamba deactivate
