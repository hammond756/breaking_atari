from breaking_atari.BaseParser import BaseParser
from breaking_atari.train import train
from breaking_atari.models.MLP import MLP
from breaking_atari.atari_wrappers.openai_wrappers import wrap_deepmind

import os
import gym

if __name__ == '__main__':

    parser = BaseParser()
    
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
    input_dims = config.grid_size[0] * config.grid_size[1] * config.n_object_types

    model = MLP(input_dims, 300, action_dims, config.sprites_dir, device=config.device)
    target = MLP(input_dims, 300, action_dims, config.sprites_dir, device=config.device)
    target.load_state_dict(model.state_dict())
    target.eval()

    train(model, target, env, config)
