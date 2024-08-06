#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 /home/zephyr/vision/DenseFusion/tools/train.py --dataset linemod\
  --dataset_root /home/zephyr/vision/DenseFusion/datasets/linemod/datasetbaru\
  --resume_posenet pose_model_2_0.007242249020685752.pth\
  --resume_refinenet pose_refine_model_current.pth
