#!/bin/bash

DATA_PATH=data

mkdir -p "$DATA_PATH/pt_data"
mkdir -p "$DATA_PATH/ft_data"
mkdir -p "$DATA_PATH/few_ft_data"
mkdir -p "$DATA_PATH/test_data"

# pretrain
wget https://huggingface.co/datasets/qiyuw/wspalign_pt_data/resolve/main/train-6langs.json \
  -O "$DATA_PATH/pt_data/train-6langs.json"

wget https://huggingface.co/datasets/qiyuw/wspalign_pt_data/resolve/main/kftt_dev.json \
  -O "$DATA_PATH/pt_data/kftt_dev.json"

# finetune
for LANG in kftt deen enfr roen
do
    wget "https://huggingface.co/datasets/qiyuw/wspalign_ft_data/resolve/main/${LANG}_ft.json" \
      -O "$DATA_PATH/ft_data/${LANG}_ft.json"
done

# few shot
for LANG in kftt deen enfr roen
do
    wget "https://huggingface.co/datasets/qiyuw/wspalign_few_ft_data/resolve/main/${LANG}_few.json" \
      -O "$DATA_PATH/few_ft_data/${LANG}_few.json"
done

# test and eval dataset
for LANG in kftt deen enfr roen
do
    wget "https://huggingface.co/datasets/qiyuw/wspalign_test_data/resolve/main/${LANG}_test.json" \
      -O "$DATA_PATH/test_data/${LANG}_test.json"
done