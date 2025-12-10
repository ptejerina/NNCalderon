#!/bin/bash
#SBATCH --job-name=TestAli
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=gn04,gn05,gn06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=5-00:00:00
#SBATCH --partition=unlimited
#SBATCH --output=/home/akalout/INV_GAUSSIAN_FIXED/results/job%j.out
#SBATCH --error=/home/akalout/INV_GAUSSIAN_FIXED/results/job%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alikalout91@gmail.com

# safer shell, but avoid unbound-var crashes by guarding vars
set -eo pipefail

# 1) activate your venv
source /home/akalout/ali_env/bin/activate

# 2) headless plotting + unbuffered logs
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

# 3) go to submit dir
cd "$SLURM_SUBMIT_DIR"

# 4) make local imports work (guard PYTHONPATH)
export PYTHONPATH="${PYTHONPATH:-}:$SLURM_SUBMIT_DIR"

python3 run_inv_gaussian_waveletsBC.py
