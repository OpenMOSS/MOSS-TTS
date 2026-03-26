#!/usr/bin/env bash
set -euo pipefail

nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /root/moss-tts-venv/bin/python /root/moss-tts-norwegian/train_lora.py \
  --gpu 0 \
  --trainable-lora-modules mlp \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lm-heads-mode none \
  --lr 2e-6 \
  --warmup-steps 100 \
  --max-train-steps 30000 \
  --save-steps 500 \
  --eval-steps 500 \
  --log-steps 10 \
  --weight-decay 0.01 \
  --max-grad-norm 0.5 \
  --wandb-name moss-scratch-long-C-mlp-r16-lr2e6-s500 \
  --output-dir /root/moss-tts-norwegian/checkpoints_scratch_long_C_mlp_r16 \
  > /root/moss-tts-norwegian/scratch_long_C_mlp_r16.log 2>&1 &
