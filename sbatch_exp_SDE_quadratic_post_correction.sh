#!/bin/bash
#SBATCH --job-name=dgdlm_array
#SBATCH --output=/home/c02sara/CISPA-az6/dgdlm-2024/higher_interpolant_matching/tmp_out_folder/job-%A_%a.out
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=tmp,gpu,xe8545
#SBATCH --time=10000
#SBATCH --exclude=xe8545-a100-06,xe8545-a100-11
#SBATCH --array=1-96

set -euo pipefail

CMD_FILE=/home/c02sara/CISPA-az6/dgdlm-2024/higher_interpolant_matching/SDE_quadratic_post_correction.txt
NUM_CFGS=$(wc -l < "$CMD_FILE")
NUM_REPEATS=3

# Map array task ID to (config_idx, repeat_num)
cfg_idx=$(( (SLURM_ARRAY_TASK_ID-1) / NUM_REPEATS + 1 ))
repeat_num=$(( (SLURM_ARRAY_TASK_ID-1) % NUM_REPEATS + 1 ))

ARGS="$(sed -n "${cfg_idx}p" "$CMD_FILE")"

JOBDATADIR=$(ws create work --space "$SLURM_JOB_ID" --duration "7 00:00:00")
JOBTMPDIR=/tmp/job-"$SLURM_JOB_ID"

srun mkdir -p "$JOBDATADIR" "$JOBTMPDIR"

echo "Task ${SLURM_ARRAY_TASK_ID}: Config ${cfg_idx}, Repeat ${repeat_num}"
echo "python main.py $ARGS"
echo "Job Data Dir: $JOBDATADIR"
echo "Job tmp dir: $JOBTMPDIR"

srun --container-image=projects.cispa.saarland:5005#c02sara/dynamic_and_geometric_models:torch_flow_matching_v17 \
     --container-mounts="${JOBTMPDIR}:/tmp" \
     python /home/c02sara/CISPA-az6/dgdlm-2024/higher_interpolant_matching/main.py $ARGS