#!/bin/bash

python -u train_baseline.py --num_frames 500000 --environment PongNoFrameskip-v4 --output_dir $TMPDIR/$1 --image_size 84 84 --eps_start 1.0 --eps_stop 0.02 --eps_steps 100000 --target_update 1000 --memory 100000 --gamma 0.99 --lr 0.0001 --batch_size 128 --optimize_every 4 --exploration_phase 10000 --device $2

cp -r $TMPDIR/$1 "$HOME/breaking_atari/jobs/$1"
