#!/bin/bash

set -e

DATA_PATH=data

mkdir -p "$DATA_PATH/pt_data"

echo "Creating small pretraining datasets..."

python create_small_dataset.py \
    --input "$DATA_PATH/pt_data/train-6langs.json" \
    --output "$DATA_PATH/pt_data/train-6langs-small.json" \
    --max_qas 500

python create_small_dataset.py \
    --input "$DATA_PATH/pt_data/kftt_dev.json" \
    --output "$DATA_PATH/pt_data/kftt_dev-small.json" \
    --max_qas 100

echo ""
echo "Done!"
echo "Created:"
echo "  $DATA_PATH/pt_data/train-6langs-small.json"
echo "  $DATA_PATH/pt_data/kftt_dev-small.json"