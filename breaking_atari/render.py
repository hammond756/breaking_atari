import argparse
import gym
import torch
from breaking_atari.atari_wrappers.openai_wrappers import wrap_deepmind
from breaking_atari.utils import select_action

def import_class(name):
    module, class_name = name.rsplit('.', 1)
    mod = __import__(module, fromlist=[class_name])
    mod = getattr(mod, class_name)
    return mod

def render(model, env):

    obs = env.reset()

    done = False
    while not done:
        env.unwrapped.render()
        action = select_action(model, obs, env.action_space.n, 0)
        next_obs, reward, done, _ = env.step(action.item())

        obs = next_obs



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_class', type=str, required=True)
    parser.add_argument('--parameters', type=str, required=False)
    parser.add_argument('--image_size', type=int, nargs=2, required=True)
    parser.add_argument('--environment', type=str, required=True)
    parser.add_argument('--model_config', type=int, nargs='+', required=True)

    config = parser.parse_args()

    env = gym.make(config.environment)
    env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, warp=config.image_size)

    ModelClass = import_class(config.model_class)
    print(ModelClass)

    model = ModelClass(*config.model_config)

    for ep in range(1,53):
        state_dict = torch.load('../baseline_results/baseline_results/dqn-{}'.format(ep), map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        render(model, env)