#!/bin/bash

if [ ! $# -eq 3 ]
then
    echo "Give 3 arguments"
    exit
fi

mkdir $TMPDIR/$1

python -u train_handcrafted.py --num_frames 50000 --environment SpaceInvadersDeterministic-v4 --output_dir $TMPDIR/$1 --sprites_dir "$PWD/sprites/space_invaders/" --eps_start 1.0 --eps_stop 0.1 --eps_steps 10000 --target_update 1000 --memory 10000 --gamma 0.99 --lr 0.0001 --batch_size 32 --optimize_every 1 --exploration_phase 5000 --num_eval 1000 --eval_every 1000 --device $2

cp -r $TMPDIR$1 "$PWD/$1"
