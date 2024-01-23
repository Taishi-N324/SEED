#!/bin/bash -x

#SBATCH --account=cstdl
#SBATCH --nodes=1
#SBATCH --output=logs/out.%j
#SBATCH --error=error_logs/err.%j
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --cpus-per-task=96

export CUDA_VISIBLE_DEVICES=0,1,2,3 # ensures GPU_IDs are available with correct indicies

# Args
START_SHARD="0020001"
echo START_SHARD=$START_SHARD

END_SHARD="0021500"
echo END_SHARD=$END_SHARD

PATHS="/p/fastdata/mmlaion/datacomp/datacomp_1B/flat/{$START_SHARD..$END_SHARD}.tar"
echo PATHS=$PATHS

OUTPUT_DIR="/p/fastdata/mmlaion/seed_tokens_datacomp1b_20_to_30/"
echo OUTPUT_PATH=$OUTPUT_DIR

NUM_WORKERS=48
echo NUM_WORKERS=$NUM_WORKERS

NUM_GPUS=4
echo NUM_GPUS=$NUM_GPUS

BATCH_SIZE=2048
echo BATCH_SIZE=$BATCH_SIZE

# Args
module load Stages/2023 GCC/11.3.0  OpenMPI/4.1.4
ml git

source /p/project/ccstdl/gupta6/miniconda3/bin/activate
conda activate seed

srun --cpu-bind=v --accel-bind=gn python -u seed_tokens.py -p $PATHS \
						-o $OUTPUT_DIR \
						-nw $NUM_WORKERS \
						-ng $NUM_GPUS \
						-bs $BATCH_SIZE

# python -u : produce output immediately, no buffer caching
#srun --cpu-bind=v --accel-bind=gn  python -u dummy_script.py
