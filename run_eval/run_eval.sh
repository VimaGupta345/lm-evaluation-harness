#!/bin/bash

TARGET_DIR="/storage/coda1/p-apadmanabh3/0/vgupta345/lm-eval/lm_eval_new/lm-evaluation-harness/run_eval/sbatch_files_drop_2"
mkdir -p "$TARGET_DIR"

# Define the tasks and corresponding few-shot values
declare -A TASKS
TASKS["arc_challenge"]=25
TASKS["hellaswag"]=10
TASKS["truthfulqa"]=0
TASKS["mmlu"]=5
TASKS["winogrande"]=5
TASKS["gsm8k"]=5

# Base directory for your project
BASE_DIR="/storage/home/hcoda1/4/vgupta345/p-apadmanabh3-0/lm-eval/lm_eval_new/lm-evaluation-harness"

# MIXTRAL configuration path
MIXTRAL_CONFIG_PATH="/storage/coda1/p-apadmanabh3/0/vgupta345/prowl/mixtral_configs/none-dynamic-drop-2.json"
# Extract the last component using parameter expansion
CONFIG_NAME="${MIXTRAL_CONFIG_PATH##*/}"

# Timestamp for creating a unique directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Loop over each task and create a specific SLURM batch file
for TASK in "${!TASKS[@]}"; do
    FEWSHOT=${TASKS[$TASK]}
    FILE_NAME="${TARGET_DIR}/lm_eval_${TASK}.sbatch"
    LOG_DIR="${BASE_DIR}/logs/${TASK}/${TIMESTAMP}_${CONFIG_NAME}"  # Directory for logs
    mkdir -p "$LOG_DIR"  # Ensure the directory exists

    cat <<EOF >$FILE_NAME
#!/bin/bash
#SBATCH -J LM_eval_${TASK}                                  # Job name
#SBATCH -A gts-ag117-prism
#SBATCH -q embers
#SBATCH -N1 --gres=gpu:H100:2                               # Number of nodes and GPUs
#SBATCH --mem-per-gpu=80G                                   # Memory per GPU
#SBATCH -t8:00:00                                           # Max time (8 hours)
#SBATCH -o ${LOG_DIR}/${TASK}_%j.out                        # Output and error file
#SBATCH --mail-type=BEGIN,END,FAIL                          # Mail events
#SBATCH --mail-user=vgupta345@gatech.edu                    # Email for notifications

module load gcc/12.3.0
nvidia-smi                                                                        # validate correct config

source /storage/home/hcoda1/4/vgupta345/micromamba/etc/profile.d/micromamba.sh
micromamba activate /storage/coda1/p-apadmanabh3/0/vgupta345/prowl_new

export NCCL_P2P_DISABLE=1
export TOKENIZERS_PARALLELISM=false

cd $BASE_DIR

lm-eval --model vllm \
    --model_args pretrained="mistralai/Mixtral-8x7B-v0.1",tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.7,enforce_eager=True,mixtral_config_file="$MIXTRAL_CONFIG_PATH" \
    --tasks $TASK \
    --num_fewshot $FEWSHOT \
    --batch_size 128 \
    --wandb_args project=prowl,group=acc_drop_2,name=$TASK \
    >> "${LOG_DIR}/lm_eval_${TASK}.log"

micromamba deactivate
EOF
    echo "Created batch file for task: $TASK with few-shot: $FEWSHOT in log directory: $LOG_DIR"
done

echo "All SLURM batch files have been created with timestamped log directories and using the specified MIXTRAL config."
