#!/bin/bash
#SBATCH --job-name=ST20CPU
#SBATCH --ntasks-per-node=20  # Utilize all CPU cores
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=5-00:00:0
#SBATCH -o /home/ptejerina/NNCalderon/results/job%j.o
#SBATCH -e /home/ptejerina/NNCalderon/results/job%j.e
#SBATCH --partition=long
#SBATCH --mail-type=ALL       # Send email on job start, end, and failure
#SBATCH --mail-user=pablo.tejerina@icc.ub.edu  # Replace with your email address

module purge
#module load cuda
source /home/ptejerina/holo_env/bin/activate

# Run the first PyTorch training script
python3 SPEED_TEST.py

wait