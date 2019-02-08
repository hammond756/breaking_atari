#!/bin/bash
cp -r "$HOME/breaking_atari/dqn/space_invader_sprites" $TMPDIR/space_invader_sprites

python ../dqn/train_handcrafted.py --environment SpaceInvadersDeterministic-v4 --output_dir $TMPDIR/$1 --sprites_dir $TMPDIR/space_invader_sprites

cp -r $TMPDIR/$1 "$HOME/breaking_atari/jobs/$1"
