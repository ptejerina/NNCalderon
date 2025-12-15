#!/bin/bash
#SBATCH --job-name=FourBOX0208
#SBATCH --ntasks-per-node=1 
#SBATCH --exclude=gn04,gn05,gn06    ### #SBATCH --nodelist=gn01                 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
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
python3 NNCalderon_BOX0208_10_fourierboard_trig_wavelet_op_lin_parab_op_lin_ricker_54_BC_ReLU_actv_Hlast.py

wait