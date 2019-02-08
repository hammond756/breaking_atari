__author__ = 'Aron'

import gym
import numpy as np
from breaking_atari.atari_wrappers.utils import mask

class AnnotateFrame(gym.ObservationWrapper):
    def __init__(self, env, templates, threshold=0.95):
        super(AnnotateFrame, self).__init__(env)

        assert len(env.observation_space.shape) == 3
        assert env.observation_space.shape[2] == 1, "AnnotateFrame requires 1-channel images"

        self.templates = templates
        self.threshold=threshold

        self.observation_space = gym.spaces.Box(low=0, high=255,
                shape=(env.height, env.width, 3), dtype=np.uint8)

    def observation(self, frame):
        observation = mask(self.templates, frame, self.threshold)
        return observation