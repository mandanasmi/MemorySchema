#!/bin/bash
#SALLOC --time=3:00:00
#SALLOC --gres=gpu:1
#SALLOC --cpus-per-task=4
#SALLOC --mem-per-cpu=4G
#SALLOC --output=tmp/slurm/slurm-%A_%a.out
#SALLOC --array=0-24:5

module --force purge
module load python/3.9
module load cuda/11.2/cudnn/8.1

GROUP_NAME="5_vars"

export WANDB_MODE=offline 
export WANDB_DIR=$SCRATCH/dag-gflownet-params/$GROUP_NAME/jobs/$SLURM_ARRAY_JOB_ID
mkdir -p $WANDB_DIR

MODEL=${1:-lingauss_diag}
NUM_SAMPLES=${2:-64}
LR=${3:-5e-4}


python -m venv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

cd $SLURM_TMPDIR
git clone git@github.com:tristandeleu/dag-gflownet-devel.git
cd dag-gflownet-devel
git checkout task/post-params-gfn

# Install all the packages
pip install --upgrade pip
pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
# Uninstall PyTorch to avoid CUDA version mismatch; PyTorch is required by
# pgmpy, but we don't use the features of pgmpy that require PyTorch
pip uninstall -y torch

# Run the training script
for i in {0..4}
do
    SEED=$(($SLURM_ARRAY_TASK_ID + $i))
    echo "Running for seed: $SEED"
    python train.py \
        --group_name $GROUP_NAME \
        --batch_size 256 \
        --lr $LR \
        --params_num_samples $NUM_SAMPLES \
        --model $MODEL \
        --seed $SEED \
        --artifact "tristandeleu_mila_01/dag-gflownet-params/er1-lingauss-d005:v0"
done
