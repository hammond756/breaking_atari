import os
import gym

from breaking_atari.BaseParser import BaseParser
from breaking_atari.train import train
from breaking_atari.models.ConvNet import ConvNet
from breaking_atari.atari_wrappers.openai_wrappers import wrap_deepmind, FrameStack, WarpFrame
from breaking_atari.atari_wrappers.cartpole_visual import CartPoleVisual

if __name__ == '__main__':

    parser = BaseParser()

    # model specific parameters
    parser.add_argument('--image_size', type=int, nargs=2, required=False, default=[110, 84])

    config = parser.parse_args()

    if not os.path.isdir(config.output_dir):
        os.mkdir(config.output_dir)

    # initialize environment
    env = gym.make(config.environment)
    if config.environment == 'CartPole-v0':

        from pyvirtualdisplay import Display
        display = Display()
        display.start()
        os.environ['DISPLAY'] = ':' + str(display.display) + '.' + str(display.screen)

        env = CartPoleVisual(env, height=config.image_size[0], width=config.image_size[1])
        env = FrameStack(env, config.frame_stack)
    else:
        env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, warp=config.image_size)

    action_dims = env.action_space.n
    height, width, channels = env.observation_space.shape

    model = ConvNet(config.frame_stack, height, width, action_dims, device=config.device)
    target = ConvNet(config.frame_stack, height, width, action_dims, device=config.device)
    target.load_state_dict(model.state_dict())
    target.eval()

    train(model, target, env, config)
