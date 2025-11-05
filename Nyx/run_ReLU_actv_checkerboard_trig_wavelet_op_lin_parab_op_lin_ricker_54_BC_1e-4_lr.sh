#!/bin/bash
#SBATCH --job-name=ReLUActv
#SBATCH --ntasks-per-node=1 
#SBATCH --nodelist=gn07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
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
python3 NNCalderon_Relu_run_checkerboard_trig_wavelet_op_lin_parab_op_lin_ricker_54_BC_lr_1e-4.py

wait