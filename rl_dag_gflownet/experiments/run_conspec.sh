#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=sbatch_out/gfnconspec_k3.%A.%a.out
#SBATCH --error=sbatch_err/gfnconspec_k3.%A.%a.err
#SBATCH --job-name=gfnconspec_k3

module load anaconda/3
module load cuda/11.1/cudnn/8.1
conda activate py38jax 

python train.py --batch_size 256 prototypes --num_samples 1000 --num_variables 5 
