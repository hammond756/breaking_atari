#!/bin/bash

# Running this script (CartPole-v0) should give the following result:
# - Very quickly obtain reward of 200
# - Don't sustain that position (instable q-learning)
# - Q-function plot should look smooth, trending to 100

if [ ! $# -eq 3 ]
then
    echo "Give 3 arguments"
    exit
fi

mkdir $TMPDIR$1

python -u $3 --num_frames 50000 --environment CartPole-v0 --output_dir $TMPDIR/$1 --eps_start 1.0 --eps_stop 0.05 --eps_steps 10000 --target_update 500 --memory 1000 --gamma 0.99 --lr 0.0005 --batch_size 128 --optimize_every 4 --exploration_phase 100 --device $2 --eval_every 1000 --num_eval 1

cp -r $TMPDIR$1 "$PWD/$1"
