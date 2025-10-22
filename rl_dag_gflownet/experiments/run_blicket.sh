#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=sbatch_out/blicket_a_disj.%A.%a.out
#SBATCH --error=sbatch_err/blicket_a_disj.%A.%a.err
#SBATCH --job-name=blicket_a_disj

module load anaconda/3
module load cuda/11.1/cudnn/8.1
conda activate py38jax 

python train.py --batch_size 128 blicket --num_samples 1000 --num_variables 4 
