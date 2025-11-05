#!/bin/bash
#SBATCH --job-name=TWOpLinPar70BC
#SBATCH --ntasks-per-node=1 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=5-00:00:0
#SBATCH -o /home/ptejerina/NNCalderon/results/job%j.o
#SBATCH -e /home/ptejerina/NNCalderon/results/job%j.e
#SBATCH --partition=unlimited
#SBATCH --mail-type=ALL       # Send email on job start, end, and failure
#SBATCH --mail-user=pablo.tejerina@icc.ub.edu  # Replace with your email address

module purge
#module load cuda
source /home/ptejerina/holo_env/bin/activate

# Run the first PyTorch training script
python3 NNCalderon_run_checkerboard_trig_wavelet_opposite_linear_parabolic_70_BC.py

wait