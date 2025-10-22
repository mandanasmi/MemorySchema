#!/bin/bash
#SBATCH --job-name=dag-gfn-erdos-5
#SBATCH --output=dgfn_output.txt
#SBATCH --error=dgfn_error.txt
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

module purge
module load python/3.9
module load/cuda/11.2/cudnn/8.1

# Create the virtual environment
python -m venv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

# Install all the packages
pip install --upgrade pip
pip install "jax[cuda]==0.4.1" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "chex==0.1.5"
pip install -r requirements.txt
# Uninstall PyTorch to avoid CUDA version mismatch; PyTorch is required by
# pgmpy, but we don't use the features of pgmpy that require PyTorch
pip uninstall -y torch


python train.py --batch_size 256 erdos_renyi_lingauss --num_variables 5 --num_edges 5 --num_samples 100

