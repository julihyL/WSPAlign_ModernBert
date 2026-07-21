#!/bin/bash

# Stop immediately when a command fails.
# pipefail ensures that a Python error is not hidden by `tee`.
set -euo pipefail

# Always run from the repository root, regardless of where the script is called.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROJECT_DIR="wspalign"

EXPERIMENT_DIR="experiments"
OUTPUT_DIR="$EXPERIMENT_DIR/pretraining_small"
LOG_FILE="$EXPERIMENT_DIR/pretraining_small.log"

DATA_DIR="data/pt_data"
TRAIN_FILE="$DATA_DIR/train-6langs-small.json"
DEV_FILE="$DATA_DIR/kftt_dev-small.json"

MODEL_TYPE="modernbert"
MODEL_NAME="answerdotai/ModernBERT-base"

date
hostname
echo "Experiment directory: $EXPERIMENT_DIR"

echo ""
echo "### Small pretraining ###"

mkdir -p "$OUTPUT_DIR"

# Delete old feature caches before every run.
# This does not delete model checkpoints or prediction results.
echo ""
echo "Removing old feature caches from $OUTPUT_DIR ..."

find "$OUTPUT_DIR" \
    -maxdepth 1 \
    -type f \
    -name 'cached_*' \
    -print \
    -delete

echo "Cache cleanup completed."
echo ""

python "$PROJECT_DIR/run_spanpred.py" \
    --model_type "$MODEL_TYPE" \
    --model_name_or_path "$MODEL_NAME" \
    --do_train \
    --do_eval \
    --train_file "$TRAIN_FILE" \
    --predict_file "$DEV_FILE" \
    --learning_rate 1e-6 \
    --per_gpu_train_batch_size 2 \
    --per_gpu_eval_batch_size 2 \
    --num_train_epochs 1 \
    --max_seq_length 384 \
    --max_query_length 158 \
    --max_answer_length 158 \
    --doc_stride 64 \
    --n_best_size 10 \
    --data_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir \
    --overwrite_cache \
    --save_steps 100000 \
    --logging_steps 10 \
    --threads 4 \
    --version_2_with_negative \
    --warmup_steps 0 \
    2>&1 | tee "$LOG_FILE"