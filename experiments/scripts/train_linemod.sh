#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 /home/zephyr/vision/DenseFusion/tools/train.py --dataset linemod\
  --dataset_root /home/zephyr/vision/DenseFusion/datasets/linemod/datasetbaru
