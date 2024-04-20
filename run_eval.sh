#!/bin/bash

# Define the tasks and corresponding few-shot values
declare -A TASKS
TASKS["arc-challenge"]=25
TASKS["hellaswag"]=10
TASKS["truthfulqa-mc"]=0
TASKS["mmlu"]=5
TASKS["winogrande"]=5
TASKS["gsm8k"]=5

# Base directory for your project
BASE_DIR="/storage/home/hcoda1/4/vgupta345/p-apadmanabh3-0/lm-eval/lm_eval_new/lm-evaluation-harness"

# Loop over each task and create a specific SLURM batch file
for TASK in "${!TASKS[@]}"; do
    FEWSHOT=${TASKS[$TASK]}
    FILE_NAME="lm_eval_${TASK}.sh"
    cat <<EOF >$FILE_NAME
#!/bin/bash
#SBATCH -J LM_eval_${TASK}                                  # Job name
#SBATCH -A gts-ag117
#SBATCH -q embers
#SBATCH -N1 --gres=gpu:A100:1                               # Number of nodes and GPUs
#SBATCH --mem-per-gpu=80G                                   # Memory per GPU
#SBATCH -t6:00:00                                           # Max time (6 hours)
#SBATCH -o ${TASK}_%j.out                                   # Output and error file
#SBATCH --mail-type=BEGIN,END,FAIL                          # Mail events
#SBATCH --mail-user=vgupta345@gatech.edu                    # Email for notifications

nvidia-smi  # Validate GPU configuration

source /storage/home/hcoda1/4/vgupta345/micromamba/etc/profile.d/micromamba.sh
micromamba activate /storage/coda1/p-apadmanabh3/0/vgupta345/prowl_new

cd $BASE_DIR

lm-eval --model vllm \
    --model_args pretrained="mistralai/Mixtral-8x7B-v0.1",tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.7,enforce_eager=True,mixtral_config_file="\$MIXTRAL_CONFIG_PATH" \
    --tasks $TASK \
    --num_fewshot $FEWSHOT \
    --batch_size auto \
    --wandb_args project=prowl,group=$TASK \
    >> "lm_eval_${TASK}.log"

micromamba deactivate
EOF
    echo "Created batch file for task: $TASK with few-shot: $FEWSHOT"
done

echo "All SLURM batch files have been created."
