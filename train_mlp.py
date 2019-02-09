from breaking_atari.BaseParser import BaseParser
from breaking_atari.models.MLP import MLP
from breaking_atari.train import train

import gym

if __name__ == '__main__':
    parser = BaseParser()
    config = parser.parse_args()

    env = gym.make(config.environment)

    model = MLP(env.observation_space.shape[0], 200, env.action_space.n, config.device)
    target = MLP(env.observation_space.shape[0], 200, env.action_space.n, config.device)
    target.load_state_dict(model.state_dict())
    target.eval()

    train(model, target, env, config)