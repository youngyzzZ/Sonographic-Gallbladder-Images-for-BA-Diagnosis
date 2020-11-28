#!/usr/bin/env bash
set -e
cd ..
export PYTHONPATH=`pwd`:$PYTHONPATH

WORK_DIR=$(pwd)
SRC_DIR="${WORK_DIR}/src"

python "${SRC_DIR}"/main.py train\
  --desc='train_contain_empty_color_sgd_5' \
  --cuda=0 \
  --dataset='Gallbladder'\
  --model='se_resnet' \
  --action='base' \
  --epoch=210 \
  --batch_size=16 \
  --img_size=224 \
  --train_csv='label_contain_empty/label_contain_empty_test_5_all.csv' \
  --test_csv='label_contain_empty/label_contain_empty_test_5_all.csv' \
  --personal_eval \
  --optimizer='sgd' \


# 不同数据集要去调整weight