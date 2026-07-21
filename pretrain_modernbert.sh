#!/bin/bash

set -euo pipefail

# Always run from the directory containing this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROJECT_DIR="wspalign"

EXPERIMENT_DIR="experiments"
OUTPUT_DIR="$EXPERIMENT_DIR/modernbert_pretraining"
LOG_FILE="$EXPERIMENT_DIR/modernbert_pretraining.log"

DATA_DIR="data"
TRAIN_FILE="$DATA_DIR/train-6langs.json"
DEV_FILE="$DATA_DIR/kftt_dev.json"

MODEL_TYPE="modernbert"
BASE_MODEL="answerdotai/ModernBERT-base"

# Select physical GPU.
# Example:
# GPU_ID=4 bash pretrain_modernbert.sh
GPU_ID="${GPU_ID:-4}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

mkdir -p "$EXPERIMENT_DIR"
mkdir -p "$OUTPUT_DIR"

date
hostname

echo "Experiment directory: $EXPERIMENT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Training file: $TRAIN_FILE"
echo "Development file: $DEV_FILE"
echo "Physical GPU selected: $GPU_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo ""
echo "### ModernBERT pretraining ###"
echo ""

# Check dataset files.
if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "ERROR: Training file not found: $TRAIN_FILE" >&2
    exit 1
fi

if [[ ! -f "$DEV_FILE" ]]; then
    echo "ERROR: Development file not found: $DEV_FILE" >&2
    exit 1
fi

# Find the newest complete checkpoint.
LATEST_CHECKPOINT=""

while IFS= read -r CHECKPOINT; do
    if [[ -f "$CHECKPOINT/config.json" \
        && -f "$CHECKPOINT/model.safetensors" \
        && -f "$CHECKPOINT/optimizer.pt" \
        && -f "$CHECKPOINT/scheduler.pt" ]]; then

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
    MODEL_NAME="$LATEST_CHECKPOINT"

    echo "Complete checkpoint found:"
    echo "  $LATEST_CHECKPOINT"
    echo "Training will resume from this checkpoint."
else
    MODEL_NAME="$BASE_MODEL"

    echo "No complete checkpoint found."
    echo "Training will start from:"
    echo "  $BASE_MODEL"
fi

echo ""

# Confirm CUDA before starting.
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Refusing to start CPU training."
    )

print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count:", torch.cuda.device_count())
print("Logical CUDA device:", torch.cuda.current_device())
print("GPU name:", torch.cuda.get_device_name(0))
print("PyTorch CUDA version:", torch.version.cuda)
PY

echo ""
echo "Starting or resuming training..."
echo ""

# Add a separator instead of overwriting the previous log.
{
    echo ""
    echo "============================================================"
    echo "Run started: $(date)"
    echo "Model source: $MODEL_NAME"
    echo "Physical GPU: $GPU_ID"
    echo "============================================================"
} | tee -a "$LOG_FILE"

python "$PROJECT_DIR/run_spanpred.py" \
    --model_type "$MODEL_TYPE" \
    --model_name_or_path "$MODEL_NAME" \
    --do_train \
    --do_eval \
    --train_file "$TRAIN_FILE" \
    --predict_file "$DEV_FILE" \
    --learning_rate 1e-5 \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --max_seq_length 1024 \
    --max_query_length 158 \
    --max_answer_length 158 \
    --doc_stride 128 \
    --n_best_size 10 \
    --data_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir \
    --save_steps 250 \
    --logging_steps 100 \
    --threads 1 \
    --version_2_with_negative \
    --warmup_steps 200 \
    2>&1 | tee -a "$LOG_FILE"