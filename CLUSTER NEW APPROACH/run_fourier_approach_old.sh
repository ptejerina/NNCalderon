#!/bin/bash
#SBATCH --job-name=TestAli
#SBATCH --ntasks-per-node=1  # Utilize all CPU cores
#SBATCH --exclude=gn04,gn05,gn06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=5-00:00:0
#SBATCH -o /home/akalout/NEWAPPROACH/results/job%j.o
#SBATCH -e /home/akalout/NEWAPPROACH/results/job%j.e
#SBATCH --partition=unlimited
#SBATCH --mail-type=ALL       # Send email on job start, end, and failure
#SBATCH --mail-user=alikalout91@gmail.com  # Replace with your email address

module purge
#module load cuda
source /home/akalout/ali_env/bin/activate

# Run the first PyTorch training script
python3 run_new_approach.py

wait