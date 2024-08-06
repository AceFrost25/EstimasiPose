#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 /home/zephyr/vision/DenseFusion/tools/eval_linemod_on_image.py --dataset_root /home/zephyr/vision/DenseFusion/datasets/linemod/datasetbaru\
  --model /home/zephyr/vision/DenseFusion/trained_models/linemod/pose_model_3_0.008452475054865263.pth\
  --refine_model /home/zephyr/vision/DenseFusion/trained_models/linemod/pose_refine_model_8_0.008304940117657141.pth