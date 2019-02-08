#!/bin/bash

python -u train_baseline.py --num_frames 500000 --environment PongNoFrameskip-v4 --output_dir $TMPDIR/$1 --eps_start 1.0 --eps_stop 0.02 --eps_steps 100000 --target_update 1000 --memory 100000 --gamma 0.99 --lr 0.0001 --batch_size 32 --optimize_every 1 --exploration_phase 0

cp -r $TMPDIR/$1 "$HOME/breaking_atari/jobs/$1"
