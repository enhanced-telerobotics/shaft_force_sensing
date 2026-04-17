#!/bin/bash

#SBATCH --job-name=force_sensing        # job name
#SBATCH -p gpu                          # partition (queue)
#SBATCH --gres=gpu:1                    # number of GPUs requested
#SBATCH --nodelist=gput[053-075]        # allowed node range
#SBATCH --exclude=gput[054-056]         # excluded bad node
#SBATCH -N 1                            # total number of nodes requested
#SBATCH -c 4                            # total number of tasks (one for each core)
#SBATCH --time=4:00:00                  # total run time limit (HH:MM:SS)
#SBATCH --mem=20gb                      # memory required per node
#SBATCH -o output/slurm-%j-%N.out       # STDOUT file
#SBATCH -e output/slurm-%j-%N.err       # STDERR file

# Load the necessary modules (if any)

# Your job command(s) here:

# Define paths
PYTHON_BIN="/home/zxc703/python_userbases/torch_stable/bin/python"
SCRATCH_DIR="/scratch/pioneer/users/sxy841/shaft_force_sensing"
PROGRAM_DIR="/home/sxy841/ERIE/shaft_force_sensing"
TEMP_DIR="${TMPDIR:-/tmp/job.${SLURM_JOB_ID}.pioneer}"

# Display the GPU information
nvidia-smi

# Display the passed arguments
echo "Arguments passed to the script: $@"

# Change to the scratch directory
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

# Ensure the checkpoints directory exists
mkdir -p "${SCRATCH_DIR}/logs"
ln -s "${SCRATCH_DIR}/logs"

# Copy data from the program directory
rsync -a "${PROGRAM_DIR}/data" ./

# Run the training program
"$PYTHON_BIN" "${PROGRAM_DIR}/shaft_force_sensing/training/trainer.py" "$@" --job_id "${SLURM_JOB_ID}"

# Then run the prediction program
"$PYTHON_BIN" "${PROGRAM_DIR}/shaft_force_sensing/training/predictor.py" "$@" --job_id "${SLURM_JOB_ID}"

# Clean up the temporary directory
rm -rf "$TEMP_DIR/*"

# Optional: Copy to NAS
# /usr/local/software/rclone/1.68/rclone copy "${SCRATCH_DIR}/checkpoints" sxy841:/checkpoints

# Display the end date
date
