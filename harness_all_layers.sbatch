#!/bin/bash
#SBATCH -J LM_eval_all_layers                                # Job name
#SBATCH -A gts-ag117
#SBATCH -q embers
#SBATCH -N1 --gres=gpu:A100:2                               # Number of nodes, GPUs, and cores required
#SBATCH --mem-per-gpu=80G                                   # Memory per gpu
#SBATCH -t6:00:00                                        # Duration of the job (Ex: 15 mins)
#SBATCH -o all_layers_%j.out                             # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL                          # Mail preferences
#SBATCH --mail-user=vgupta345@gatech.edu                     # e-mail address for notifications

module load gcc/12.3.0
nvidia-smi                                                                        # validate correct config

source /storage/home/hcoda1/4/vgupta345/micromamba/etc/profile.d/micromamba.sh
micromamba activate /storage/coda1/p-apadmanabh3/0/vgupta345/prowl_new/

cd /storage/coda1/p-apadmanabh3/0/vgupta345/lm-eval/lm-evaluation-harness/

MODEL_ARGS="pretrained=/storage/coda1/p-apadmanabh3/0/vgupta345/merging_exp/slerp_0.5_Wizard7B_ShishaGamma7B/all_layers/"
TASK="mgsm-1.0-0.6,gsm8k"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot "5,5" \
    --batch_size 100 \
    --device "cuda" \
    --output_path "result_all.json"

micromamba deactivate
