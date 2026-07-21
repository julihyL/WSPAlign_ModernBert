#!/usr/bin/env bash

set -euo pipefail

# Always run from the directory containing this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROJECT_DIR="wspalign"

EXPERIMENT_DIR="experiments"
OUTPUT_DIR="$EXPERIMENT_DIR/modernbert_384_10epochs_2gpu"
LOG_FILE="$EXPERIMENT_DIR/modernbert_384_10epochs_2gpu.log"

DATA_DIR="data/pt_data"
TRAIN_FILE="$DATA_DIR/train-6langs.json"
DEV_FILE="$DATA_DIR/kftt_dev.json"

MODEL_TYPE="modernbert"
BASE_MODEL="answerdotai/ModernBERT-base"

# Physical GPU IDs can be overridden when launching the script.
#
# Example:
# GPU_IDS=4,5 bash pretrain_modernbert_384_2gpu.sh
GPU_IDS="${GPU_IDS:-4,5}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

# Effective batch size:
# 4 samples per GPU × 2 GPUs × 16 accumulation steps = 128
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"

LEARNING_RATE="${LEARNING_RATE:-1e-6}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-50}"

mkdir -p "$EXPERIMENT_DIR"
mkdir -p "$OUTPUT_DIR"

date
hostname

echo "============================================================"
echo "ModernBERT 384-token training"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Training file: $TRAIN_FILE"
echo "Development file: $DEV_FILE"
echo "Physical GPUs: $GPU_IDS"
echo "Per-GPU batch size: $PER_GPU_BATCH_SIZE"
echo "Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Effective batch size: $((PER_GPU_BATCH_SIZE * 2 * GRADIENT_ACCUMULATION_STEPS))"
echo "Number of epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Warmup steps: $WARMUP_STEPS"
echo "============================================================"

# Stop immediately if the training dataset is missing.
if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "ERROR: Training file not found: $TRAIN_FILE" >&2
    exit 1
fi

# Stop immediately if the development dataset is missing.
if [[ ! -f "$DEV_FILE" ]]; then
    echo "ERROR: Development file not found: $DEV_FILE" >&2
    exit 1
fi

# Find the newest complete checkpoint in the output directory.
LATEST_CHECKPOINT=""

while IFS= read -r CHECKPOINT; do
    if [[ -f "$CHECKPOINT/config.json" \
        && -f "$CHECKPOINT/optimizer.pt" \
        && -f "$CHECKPOINT/scheduler.pt" \
        && ( -f "$CHECKPOINT/model.safetensors" \
             || -f "$CHECKPOINT/pytorch_model.bin" ) ]]; then

        LATEST_CHECKPOINT="$CHECKPOINT"
        break
    fi
done < <(
    find "$OUTPUT_DIR" \
        -maxdepth 1 \
        -type d \
        -name 'checkpoint-*' \
        | sort -Vr
)

if [[ -n "$LATEST_CHECKPOINT" ]]; then
    MODEL_SOURCE="$LATEST_CHECKPOINT"

    echo "Resuming from checkpoint:"
    echo "  $MODEL_SOURCE"
else
    MODEL_SOURCE="$BASE_MODEL"

    echo "No complete checkpoint was found."
    echo "Starting from the base model:"
    echo "  $MODEL_SOURCE"
fi

# Confirm that exactly two CUDA devices are visible.
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Refusing to run training on CPU."
    )

gpu_count = torch.cuda.device_count()

print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count:", gpu_count)
print("PyTorch CUDA version:", torch.version.cuda)

for index in range(gpu_count):
    print(
        f"Logical cuda:{index}: "
        f"{torch.cuda.get_device_name(index)}"
    )

if gpu_count != 2:
    raise RuntimeError(
        f"Expected exactly 2 visible GPUs, but found {gpu_count}."
    )
PY

# Append run information instead of overwriting the existing log.
{
    echo ""
    echo "============================================================"
    echo "Run started: $(date)"
    echo "Model source: $MODEL_SOURCE"
    echo "Physical GPUs: $GPU_IDS"
    echo "Maximum sequence length: 384"
    echo "Number of epochs: $NUM_EPOCHS"
    echo "Per-GPU batch size: $PER_GPU_BATCH_SIZE"
    echo "Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
    echo "============================================================"
} | tee -a "$LOG_FILE"

python "$PROJECT_DIR/run_spanpred.py" \
    --model_type "$MODEL_TYPE" \
    --model_name_or_path "$MODEL_SOURCE" \
    --do_train \
    --do_eval \
    --train_file "$TRAIN_FILE" \
    --predict_file "$DEV_FILE" \
    --learning_rate "$LEARNING_RATE" \
    --per_gpu_train_batch_size "$PER_GPU_BATCH_SIZE" \
    --per_gpu_eval_batch_size 8 \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --num_train_epochs "$NUM_EPOCHS" \
    --max_seq_length 384 \
    --max_query_length 158 \
    --max_answer_length 158 \
    --doc_stride 64 \
    --n_best_size 10 \
    --data_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir \
    --save_steps "$SAVE_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --threads 4 \
    --version_2_with_negative \
    --warmup_steps "$WARMUP_STEPS" \
    2>&1 | tee -a "$LOG_FILE"