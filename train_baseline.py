import os
import gym
import argparse

from breaking_atari.train import train
from breaking_atari.models.ConvNet import ConvNet
from breaking_atari.atari_wrappers.openai_wrappers import wrap_deepmind

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # defaults come from Mnih et al. (2015)
    parser.add_argument('--batch_size', type=int, required=False, default=32)
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
    parser.add_argument('--optimize_every', type=int, required=False, default=4)
    parser.add_argument('--device', type=str, required=True)

    # model specific parameters
    parser.add_argument('--image_size', type=int, nargs=2, required=False, default=[110, 84])

    config = parser.parse_args()

    if not os.path.isdir(config.output_dir):
        os.mkdir(config.output_dir)

    # initialize environment
    env = gym.make(config.environment)
    env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, warp=config.image_size)

    action_dims = env.action_space.n
    height, width, channels = env.observation_space.shape

    model = ConvNet(config.frame_stack, height, width, action_dims, device=config.device)
    target = ConvNet(config.frame_stack, height, width, action_dims, device=config.device)
    target.load_state_dict(model.state_dict())
    target.eval()

    train(model, target, env, config)
