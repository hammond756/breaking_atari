from breaking_atari.train import train
from breaking_atari.BaseParser import BaseParser
from breaking_atari.models.ConvNet import ConvNet
from breaking_atari.atari_wrappers.utils import Template
from breaking_atari.atari_wrappers.annotate_wrapper import AnnotateFrame
from breaking_atari.atari_wrappers.openai_wrappers import wrap_deepmind

import os
import gym

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

    parser = BaseParser()
    
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
