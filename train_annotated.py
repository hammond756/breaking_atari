from breaking_atari.train import train
from breaking_atari.models.ConvNet import ConvNet
from breaking_atari.atari_wrappers.utils import Template
from breaking_atari.atari_wrappers.annotate_wrapper import AnnotateFrame
from breaking_atari.atari_wrappers.openai_wrappers import wrap_deepmind

import os
import gym
import argparse

def load_templates(sprites_dir):
    enemy_templates = [
        Template(sprites_dir + 'enemy_0_a.png', 'enemy'),
        Template(sprites_dir + 'enemy_0_b.png', 'enemy'),
        Template(sprites_dir + 'enemy_1_a.png', 'enemy'),
        Template(sprites_dir + 'enemy_1_b.png', 'enemy'),
        Template(sprites_dir + 'enemy_2_a.png', 'enemy'),
        Template(sprites_dir + 'enemy_2_b.png', 'enemy'),
        Template(sprites_dir + 'enemy_3_a.png', 'enemy'),
        Template(sprites_dir + 'enemy_3_b.png', 'enemy'),
        Template(sprites_dir + 'enemy_4_a.png', 'enemy'),
        Template(sprites_dir + 'enemy_4_b.png', 'enemy'),
        Template(sprites_dir + 'enemy_5_a.png', 'enemy'),
        Template(sprites_dir + 'enemy_5_b.png', 'enemy')
    ]

    bunker_template = Template(sprites_dir + 'defense.png', 'barrier')
    agent_template = Template(sprites_dir + 'my_sprite.png', 'agent')
    bullet_template = Template(sprites_dir + 'enemy_bullet.png', 'enemy')

    return enemy_templates + [bunker_template, agent_template, bullet_template]

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
    parser.add_argument('--environment', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--optimize_every', type=str, required=False, default=4)
    parser.add_argument('--device', type=str, required=True)
    
    # model specific parameters
    parser.add_argument('--sprites_dir', type=str, required=True)

    config = parser.parse_args()

    if not os.path.isdir(config.output_dir):
        os.mkdir(config.output_dir)

    # initialize environment
    env = gym.make(config.environment)
    env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, warp=(210,160))
    env = AnnotateFrame(env, templates=load_templates(config.sprites_dir), threshold=0.95)
    
    action_dims = env.action_space.n
    height, width, channels = env.observation_space.shape

    model = ConvNet(channels, height, width, action_dims, device=config.device)
    target = ConvNet(channels, height, width, action_dims, device=config.device)
    target.load_state_dict(model.state_dict())
    target.eval()

    train(model, target, env, config)
