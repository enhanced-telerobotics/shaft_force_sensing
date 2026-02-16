#!/bin/bash
#SBATCH --job-name=force_sensing
#SBATCH -p gpu
#SBATCH -C gpul40s
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --time=1-00:00:00
#SBATCH --mem=20gb
#SBATCH -o output/slurm-%N-%j.out
#SBATCH -e output/slurm-%N-%j.err

echo "========== JOB INFO =========="
hostname
date
echo "Job ID: $SLURM_JOB_ID"
echo "==============================="

# ---- Activate conda environment ----
source ~/.bashrc
conda activate ltc311

# (Sometimes needed on clusters)
module load cuda 2>/dev/null

# Thread usage
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Python:"
which python
python -V

echo "CUDA:"
python - <<EOF
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
EOF

# ---- Paths ----
SCRATCH_DIR="/scratch/pioneer/users/sxk2514/shaft_force_sensing"
PROGRAM_DIR="/home/sxk2514/erie/shaft_force_sensing"
TEMP_DIR="${TMPDIR:-/tmp/job.${SLURM_JOB_ID}.pioneer}"

mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

# ---- Logs ----
mkdir -p "${SCRATCH_DIR}/logs"
ln -s "${SCRATCH_DIR}/logs" logs

# ---- Copy dataset locally (fast IO) ----
rsync -a "${PROGRAM_DIR}/data" ./

# ---- GPU info ----
nvidia-smi

# ---- Run training ----
python "${PROGRAM_DIR}/shaft_force_sensing/training/trainer.py" "$@" --job_id "${SLURM_JOB_ID}"

echo "Arguments passed:"
echo "$@"

# ---- Cleanup ----
rm -rf "$TEMP_DIR"

echo "Finished at:"
date



# example usage : 
# sbatch train.slurm --seed 1 --batch_size 256 --max_epochs 200 --model_type ltc --ablation_config No_Hex10