#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 /home/zephyr/vision/DenseFusion/tools/eval_linemod.py --dataset_root /home/zephyr/vision/DenseFusion/datasets/linemod/datasetbaru\
  --model /home/zephyr/vision/DenseFusion/trained_models/linemod/pose_model_2_0.007242249020685752.pth\
  --refine_model /home/zephyr/vision/DenseFusion/trained_models/linemod/pose_refine_model_6_0.007221526785482032.pth