#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 /home/zephyr/vision/DenseFusion/tools/box.py --dataset_root /home/zephyr/vision/DenseFusion/datasets/linemod/datasetbaru\
  --model /home/zephyr/vision/DenseFusion/trained_models/linemod/pose_model_2_0.0055989065872149866.pth\
  --refine_model /home/zephyr/vision/DenseFusion/trained_models/linemod/pose_refine_model_20_0.004474119651810421.pth