#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py \
--resume pretrained/flow1d_highres-e0b98d7e.pth \
--val_iters 24 \
--inference_dir demo/dogs-jump \
--output_path output/flow1d-dogs-jump
