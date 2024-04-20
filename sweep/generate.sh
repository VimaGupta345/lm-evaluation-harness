#!/bin/bash

# Directory containing your YAML configurationsrectory containing your YAML configurations
CONFIG_DIR="/storage/coda1/p-apadmanabh3/0/vgupta345/prowl_new"

# Array of techniques based on the YAML files
declare -a techniques=("all_layers" "merge_attn_all" "merge_first_4" "merge_first_8" "merge_last_4" "merge_last_8" "merge_middle_8" "merge_mlp_all")
declare -a slerp_values=("0.2" "0.7")  # Slerp values to iterate over

# Loop through each slerp value
for slerp_value in "${slerp_values[@]}"; do
    # Loop through each technique
    for technique in "${techniques[@]}"; do
        technique_wandb="${technique//-/_}"  # replace underscores with dashes for wandb args
        # Create sbatch file for each technique and slerp value
        cat <<EOF > "${technique}_slerp_${slerp_value}.sbatch"
#!/bin/bash
#SBATCH -J LM_eval_${technique}_slerp_${slerp_value}                                # Job name
#SBATCH -A gts-ag117
#SBATCH -q embers
#SBATCH -N1 --gres=gpu:A100:1                               # Number of nodes, GPUs, and cores required
#SBATCH --mem-per-gpu=80G                                   # Memory per gpu
#SBATCH -t6:00:00                                        # Duration of the job
#SBATCH -o ${technique}_slerp_${slerp_value}_%j.out                             # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL                          # Mail preferences
#SBATCH --mail-user=vgupta345@gatech.edu                     # e-mail address for notifications

nvidia-smi                                                                        # validate correct config

source /storage/home/hcoda1/4/vgupta345/micromamba/etc/profile.d/micromamba.sh
micromamba activate $CONFIG_DIR

cd /storage/coda1/p-apadmanabh3/0/vgupta345/merging_exp/mergekit

mergekit-yaml merging_experiments/slerp_${slerp_value}/slerp_${technique}.yaml ../slerp_${slerp_value}_Wizard7B_Vicuna7b/${technique}/

micromamba deactivate

micromamba activate /storage/coda1/p-apadmanabh3/0/vgupta345/new_eval

cd /storage/home/hcoda1/4/vgupta345/p-apadmanabh3-0/lm-eval/lm_eval_new/lm-evaluation-harness/

lm_eval --model vllm  \
--model_args pretrained=/storage/coda1/p-apadmanabh3/0/vgupta345/merging_exp/slerp_${slerp_value}_Wizard7B_Vicuna7b/${technique} \
--tasks mmlu,lambada_openai,gsm8k,truthfulqa \
--batch_size auto \
--wandb_args project=model_merging,name=wv-${slerp_value}-${technique_wandb} |& tee merge_${slerp_value}_${technique}_new_tasks.log

micromamba deactivate
EOF
    done
done

