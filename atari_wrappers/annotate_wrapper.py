__author__ = 'Aron'

from utils import mask

class AnnotateFrame(gym.ObservationWrapper):
    def __init__(self, env, templates, threshold=0.95):
        super(AnnotateFrame, self).__init__(env)
        self.templates = templates
        self.threshold=threshold

    def observation(self, frame):
        obervation = mask(self.templates, frame, self.threshold)
        return obervation