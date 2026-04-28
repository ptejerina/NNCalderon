#!/bin/bash
#SBATCH --job-name=gaus_raw
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/akalout/CALDERON_PINNS/Gaussian_0.2/results/job%j.out
#SBATCH --error=/home/akalout/CALDERON_PINNS/Gaussian_0.2/results/job%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alikalout91@gmail.com

set -eo pipefail

source /home/akalout/ali_env/bin/activate

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cd /home/akalout/CALDERON_PINNS/Gaussian_0.2 || exit 1
export PYTHONPATH="/home/akalout/CALDERON_PINNS/Gaussian_0.2:${PYTHONPATH:-}"

echo "PWD = $(pwd)"
echo "Python = $(which python3)"
echo "Running = /home/akalout/CALDERON_PINNS/Gaussian_0.2/run_gaussian_0.2_no_ffe.py"

python3 /home/akalout/CALDERON_PINNS/Gaussian_0.2/run_gaussian_0.2_no_ffe.py