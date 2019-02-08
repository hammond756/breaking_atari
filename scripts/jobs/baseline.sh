#!/bin/bash

python ../dqn/train_baseline.py --environment SpaceInvadersDeterministic-v4 --output_dir $TMPDIR/$1 

cp -r $TMPDIR/$1 "$HOME/breaking_atari/jobs/$1"
