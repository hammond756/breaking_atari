#!/bin/bash
cp -r "$HOME/breaking_atari/dqn/space_invader_sprites" $TMPDIR/space_invader_sprites 


python -u train_handcrafted.py --sprites_dir $TMPDIR/space_invader_sprites/ --environment SpaceInvadersDeterministic-v4 --eps_steps 10000 --target_update 1000 --num_frames 10000 --eval_every 2500 --memory 1000 --exploration_phase 500 --output_dir $TMPDIR/$1 --device $2 --optimize_every 4 --batch_size 128

cp -r $TMPDIR/$1 "$HOME/breaking_atari/jobs/$1"
