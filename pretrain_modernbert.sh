#!/bin/sh

PROJECT_DIR=wspalign
EXPERIMENT_DIR=experiments
OUTPUT_DIR=$EXPERIMENT_DIR/modernbert_pretraining

DATA_DIR=data
TRAIN_FILE=$DATA_DIR/train-6langs.json
DEV_FILE=$DATA_DIR/kftt_dev.json

MODEL_TYPE=modernbert
MODEL_NAME=answerdotai/ModernBERT-base

date
hostname
echo $EXPERIMENT_DIR

echo ""
echo "### ModernBERT pretraining ###"

mkdir -p "$OUTPUT_DIR"

python "$PROJECT_DIR/run_spanpred.py" \
    --model_type "$MODEL_TYPE" \
    --model_name_or_path "$MODEL_NAME" \
    --do_train \
    --do_eval \
    --train_file "$TRAIN_FILE" \
    --predict_file "$DEV_FILE" \
    --learning_rate 1e-5 \
    --per_gpu_train_batch_size 1 \
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
    --overwrite_cache \
    --save_steps 1000 \
    --threads 1 \
    --evaluate_during_training \
    --per_gpu_eval_batch_size 1 \
    --logging_steps 100 \
    --version_2_with_negative \
    --warmup_steps 200 \
    2>&1 | tee "$EXPERIMENT_DIR/modernbert_pretraining.log"