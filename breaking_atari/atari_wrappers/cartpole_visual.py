import gym
from gym import spaces
import cv2
import numpy as np

class CartPoleVisual(gym.Wrapper):
    def __init__(self, env, height, width):
        super(CartPoleVisual, self).__init__(env)

        assert 'CartPole' in str(env), "Can only wrap CartPole environment"

        self.env = env.unwrapped
        self.observation_space = spaces.Box(low=0, high=1, dtype=np.float32,
                                            shape=(height, width, 1))

        self.height, self.width = height, width

    def reset(self):
        self.env.reset()
        return self._get_screen()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self._get_screen(), reward, done, info


    def _get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def _get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3.
        screen = self.env.render(mode='rgb_array')
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        self.env.close() # close viewport

        # Cart is in the lower half, so strip off the top and bottom of the screen
        screen_height, screen_width = screen.shape
        screen = screen[int(screen_height*0.4):int(screen_height * 0.8),:]
        view_width = int(screen_width * 0.6)
        cart_location = self._get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, slice_range]
        screen = cv2.resize(screen, (self.width, self.height), interpolation=cv2.INTER_AREA)[:,:,None]
        return screen