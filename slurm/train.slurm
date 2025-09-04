#!/bin/bash
#SBATCH -J R1
#SBATCH --gres=gpu:4
#SBATCH -p batch_grad
#SBATCH -w ariel-v3
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A_${MODEL_NAME_SAFE}.out
#SBATCH -e logs/slurm-%A_${MODEL_NAME_SAFE}.err

if [[ "$*" == *"--help"* ]]; then
  echo "Usage: sbatch slurm/train.slurm [options]"
  echo "Options:"
  echo "  --model MODEL            Model name (default: MPX0222forHF/SQL-R1-3B)"
  echo "  --data DATA_PATH         Data path (default: data/ehrsql/train)"
  echo "  --epochs N               Number of epochs (default: 3)"
  echo "  --batch-size N           Batch size per GPU (default: 2)"
  echo "  --grad-acc N             Gradient accumulation steps (default: 4)"
  echo "  --accelerator CONFIG     Accelerator config (default: multi_gpu)"
  echo "  --reward-funcs FUNCS     Reward functions (default: execution)"
  echo "  --args \"ARGS\"          Additional arguments"
  exit 0
fi

set -x -e

source ~/.bashrc
source .venv/bin/activate
START_TIME=$(date +%s)
echo "START TIME: $(date)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Default values
MODEL="MPX0222forHF/SQL-R1-3B"
DATA_PATH="data/ehrsql/train"
EPOCHS=3
BATCH_SIZE=2
GRAD_ACC=4
ACCELERATOR="multi_gpu"
REWARD_FUNCS="execution"
ADDITIONAL_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --data)
      DATA_PATH="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --grad-acc)
      GRAD_ACC="$2"
      shift 2
      ;;
    --accelerator)
      ACCELERATOR="$2"
      shift 2
      ;;
    --reward-funcs)
      REWARD_FUNCS="$2"
      shift 2
      ;;
    --args)
      ADDITIONAL_ARGS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Distributed configuration
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=4
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${NODELIST[0]}
MASTER_PORT=6000

echo "Training Configuration:"
echo "  Model: $MODEL"
echo "  Data: $DATA_PATH"  
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACC"
echo "  Reward Functions: $REWARD_FUNCS"
echo "  Nodes: $NUM_NODES"
echo "  World Size: $WORLD_SIZE"

# Force crashing on NCCL issues
export NCCL_ASYNC_ERROR_HANDLING=1

# Build command
export CMD=" \
    src/ehr_r1/train.py \
    --model-name $MODEL \
    --data-path $DATA_PATH \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRAD_ACC \
    --reward-functions $REWARD_FUNCS \
    --use-wandb \
    --bf16 \
    $ADDITIONAL_ARGS
    "

export LAUNCHER="ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file accelerate_configs/$ACCELERATOR.yaml \
    --gradient_accumulation_steps $GRAD_ACC \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --rdzv_backend=c10d \
    --max_restarts 1 \
    --tee 3 \
    "

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --nodes=$NUM_NODES \
    --ntasks=$NUM_NODES \
    "

echo "Launching training..."
srun $SRUN_ARGS bash -c "$LAUNCHER $CMD" 2>&1

END_TIME=$(date +%s)
echo "END TIME: $(date)"
ELAPSED_SECONDS=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_SECONDS / 3600))
MINUTES=$(( (ELAPSED_SECONDS % 3600) / 60 ))
SECONDS=$((ELAPSED_SECONDS % 60))
echo "TOTAL JOB TIME: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_SECONDS} seconds)"