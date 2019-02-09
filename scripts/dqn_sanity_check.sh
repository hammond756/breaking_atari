#!/bin/bash

if [ ! $# -eq 2 ]
then
    echo "Give 2 arguments"
    exit
fi

mkdir $TMPDIR$1

python -u train_mlp.py --num_frames 500000 --environment MountainCar-v0 --output_dir $TMPDIR/$1 --eps_start 1.0 --eps_stop 0.05 --eps_steps 10000 --target_update 500 --memory 1000 --gamma 0.99 --lr 0.0001 --batch_size 32 --optimize_every 1 --exploration_phase 100 --device $2 --eval_every 2000 --num_eval 800

cp -r $TMPDIR$1 "$PWD/$1"
