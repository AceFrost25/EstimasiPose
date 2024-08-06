#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 /home/zephyr/vision/DenseFusion/tools/eval_linemodcoba.py --dataset_root /home/zephyr/vision/DenseFusion/datasets/linemod/datasetbaru\
  --model /home/zephyr/vision/DenseFusion/trained_models/linemod/pose_model_5_0.012387276037135025.pth\
  --refine_model /home/zephyr/vision/DenseFusion/trained_models/linemod/pose_refine_model_23_0.007432322497072729.pth