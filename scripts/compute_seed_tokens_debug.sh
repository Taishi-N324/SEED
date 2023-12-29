#!/bin/bash -x

#SBATCH --account=cstdl
#SBATCH --nodes=1
#SBATCH --output=logs/out.%j
#SBATCH --error=error_logs/err.%j
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=develbooster
#SBATCH --cpus-per-task=96

export CUDA_VISIBLE_DEVICES=0,1,2,3 # ensures GPU_IDs are available with correct indicies

# Args
START_SHARD="00000" # For debugging
echo START_SHARD=$START_SHARD

END_SHARD="00004" # For debugging
echo END_SHARD=$END_SHARD

PATHS="/p/fastdata/mmlaion/CC-3M/{$START_SHARD..$END_SHARD}.tar"
echo PATHS=$PATHS

OUTPUT_DIR="/p/scratch/ccstdl/mhatre1/seed_tokens_cc_3M/" # For debugging
echo OUTPUT_PATH=$OUTPUT_DIR

NUM_WORKERS=48
echo NUM_WORKERS=$NUM_WORKERS

NUM_GPUS=4
echo NUM_GPUS=$NUM_GPUS

BATCH_SIZE=2048
echo BATCH_SIZE=$BATCH_SIZE

# Args

source /p/project/ccstdl/gupta6/miniconda3/bin/activate
conda activate seed

python -u seed_tokens.py -p $PATHS \
						-o $OUTPUT_DIR \
						-nw $NUM_WORKERS \
						-ng $NUM_GPUS \
						-bs $BATCH_SIZE

# python -u : produce output immediately, no buffer caching
#srun --cpu-bind=v --accel-bind=gn  python -u dummy_script.py