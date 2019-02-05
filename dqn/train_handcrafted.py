from train import train
from model.dqn import MLP
from atari_wrappers import wrap_deepmind

import os
import gym
import argparse
import torch

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # defaults come from Mnih et al. (2015)
    parser.add_argument('--batch_size', type=str, required=False, default=32)
    parser.add_argument('--gamma', type=float, required=False, default=0.99)
    parser.add_argument('--eps_start', type=float, required=False, default=1.0)
    parser.add_argument('--eps_stop', type=float, required=False, default=0.1)
    parser.add_argument('--eps_steps', type=int, required=False, default=1000000)
    parser.add_argument('--target_update', type=int, required=False, default=10000)
    parser.add_argument('--num_frames', type=int, required=False, default=10000000)
    parser.add_argument('--num_eval', type=int, required=False, default=10000)
    parser.add_argument('--eval_every', type=int, required=False, default=100000)
    parser.add_argument('--lr', type=float, required=False, default=0.00025)
    parser.add_argument('--memory', type=int, required=False, default=1000000)
    parser.add_argument('--exploration_phase', type=int, required=False, default=50000)
    parser.add_argument('--frame_stack', type=int, required=False, default=4)
    parser.add_argument('--environment', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    # model specific parameters
    parser.add_argument('--grid_size', type=int, nargs=2, required=False, default=[32,42])
    parser.add_argument('--n_object_types', type=int, required=False, default=8)
    parser.add_argument('--sprites_dir', type=str, required=True)

    config = parser.parse_args()

    if not os.path.isdir(config.output_dir):
        os.mkdir(config.output_dir)

    # initialize environment
    env = gym.make(config.environment)
    env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, warp=(210,160))
    action_dims = env.action_space.n

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_dims = config.grid_size[0] * config.grid_size[1] * config.n_object_types

    model = MLP(input_dims, action_dims, config.sprites_dir, device=device)
    target = MLP(input_dims, action_dims, config.sprites_dir, device=device)
    target.load_state_dict(model.state_dict())
    target.eval()

    train(model, target, env, config)
