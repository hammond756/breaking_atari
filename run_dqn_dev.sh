python dqn/train.py --eps_steps 10000 --environment BreakoutDeterministic-v4 --target_update 1000 --num_frames=1 --num_eval 100 --eval_every 2500 --memory 1000 --exploration_phase 500 --output_dir dqn_dev_out
