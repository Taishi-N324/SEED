#!/bin/bash -x
#SBATCH --account=cstdl
#SBATCH --nodes=1
#SBATCH --output=logs/out.%j
#SBATCH --error=error_logs/err.%j
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --cpus-per-task=24

export CUDA_VISIBLE_DEVICES=0,1,2,3 # ensures GPU_IDs are available with correct indicies

# Args

START_SHARD=$1
END_SHARD=$2
echo START_SHARD=$START_SHARD


echo END_SHARD=$END_SHARD
PATHS="/p/fastdata/mmlaion/obelics_img/obelics-train-{$START_SHARD..$END_SHARD}-of-01335"
echo PATHS=$PATHS

OUTPUT_DIR=/p/fastdata/mmlaion/obelics_seed_${START_SHARD}_${END_SHARD}

mkdir -p $OUTPUT_DIR
echo OUTPUT_PATH=$OUTPUT_DIR

NUM_WORKERS=8
echo NUM_WORKERS=$NUM_WORKERS

NUM_GPUS=4
echo NUM_GPUS=$NUM_GPUS

BATCH_SIZE=2048
echo BATCH_SIZE=$BATCH_SIZE

# Args
module load Stages/2024 
module load CUDA/12 
module load GCC/12.3.0 
module load OpenMPI/4.1.5 
module load cuDNN
module load NCCL

source /p/scratch/ccstdl/nakamura2/SEED/venv/bin/activate
cd /p/scratch/ccstdl/nakamura2/SEED/scripts

# python -u seed_tokens.py -p $PATHS \
# 						-o $OUTPUT_DIR \
# 						-nw $NUM_WORKERS \
# 						-ng $NUM_GPUS \
# 						-bs $BATCH_SIZE

python seed_tokens.py \
	-p $PATHS \
	-o $OUTPUT_DIR \
	-nw $NUM_WORKERS \
	-ng $NUM_GPUS \
	-bs $BATCH_SIZE \
	--dataset obelics

# python -u : produce output immediately, no buffer caching
#srun --cpu-bind=v --accel-bind=gn  python -u dummy_script.py
