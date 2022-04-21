#!/usr/bin/env bash

# can be trained on a single 32G V100 GPU

# chairs
CHECKPOINT_DIR=checkpoints/chairs-flow1d && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=0 python main.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--batch_size 12 \
--val_dataset chairs sintel kitti \
--val_iters 12 \
--lr 4e-4 \
--image_size 368 496 \
--summary_freq 100 \
--val_freq 10000 \
--save_ckpt_freq 5000 \
--save_latest_ckpt_freq 1000 \
--num_steps 100000 \
2>&1 | tee ${CHECKPOINT_DIR}/train.log

# things
CHECKPOINT_DIR=checkpoints/things-flow1d && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=0 python main.py \
--stage things \
--resume checkpoints/chairs-flow1d/step_100000.pth \
--no_resume_optimizer \
--checkpoint_dir ${CHECKPOINT_DIR} \
--batch_size 6 \
--val_dataset sintel kitti \
--val_iters 12 \
--lr 1.25e-4 \
--image_size 400 720 \
--freeze_bn \
--summary_freq 100 \
--val_freq 10000 \
--save_ckpt_freq 5000 \
--save_latest_ckpt_freq 1000 \
--num_steps 100000 \
2>&1 | tee ${CHECKPOINT_DIR}/train.log

# sintel
CHECKPOINT_DIR=checkpoints/sintel-flow1d && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=0 python main.py \
--stage sintel \
--resume checkpoints/things-flow1d/step_100000.pth \
--no_resume_optimizer \
--checkpoint_dir ${CHECKPOINT_DIR} \
--batch_size 6 \
--val_dataset sintel kitti \
--val_iters 12 \
--lr 1.25e-4 \
--weight_decay 1e-5 \
--gamma 0.85 \
--image_size 368 960 \
--freeze_bn \
--summary_freq 100 \
--val_freq 10000 \
--save_ckpt_freq 5000 \
--save_latest_ckpt_freq 1000 \
--num_steps 100000 \
2>&1 | tee ${CHECKPOINT_DIR}/train.log

# kitti
CHECKPOINT_DIR=checkpoints/kitti-flow1d && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=0 python main.py \
--stage kitti \
--resume checkpoints/sintel-flow1d/step_100000.pth \
--no_resume_optimizer \
--checkpoint_dir ${CHECKPOINT_DIR} \
--batch_size 6 \
--val_dataset kitti \
--val_iters 12 \
--lr 1e-4 \
--weight_decay 1e-5 \
--gamma 0.85 \
--image_size 320 1024 \
--freeze_bn \
--summary_freq 100 \
--val_freq 10000 \
--save_ckpt_freq 5000 \
--save_latest_ckpt_freq 1000 \
--num_steps 50000 \
2>&1 | tee ${CHECKPOINT_DIR}/train.log

