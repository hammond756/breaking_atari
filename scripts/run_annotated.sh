#!/bin/bash

mkdir $TMPDIR/$1

python -u train_annotated.py --num_frames 1000000 --environment SpaceInvadersDeterministic-v4 --output_dir $TMPDIR/$1 --sprites_dir "$PWD/sprites/space_invaders/" --eps_start 1.0 --eps_stop 0.1 --eps_steps 200000 --target_update 10000 --memory 100000 --gamma 0.99 --lr 0.0001 --batch_size 128 --optimize_every 4 --exploration_phase 5000 --num_eval 1 --eval_every 10000 --device $2

cp -r $TMPDIR$1 "$PWD/$1"