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
  --train_csv='label.csv' \
  --test_csv='label.csv' \
  --personal_eval \
  --optimizer='sgd' \
