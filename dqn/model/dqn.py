__author__ = 'Aron'

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from cv2 import matchTemplate, TM_CCORR_NORMED, COLOR_BGR2GRAY, cvtColor, imread, addWeighted
from .utils import transform_observation

class DQN(nn.Module):

    def __init__(self, h, w, action_dims, device='cpu'):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32

        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 200),
            nn.ReLU(),
            nn.Linear(200, action_dims)
        )

        self.h, self.w = h, w

        self.device = device
        self.to(device)

    def prepare_input(self, x):
        return transform_observation(x, (self.h, self.w))

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.fc(x.view(x.size(0), -1))


def tile(coords, grid):
    new_y = np.floor(coords[0] / grid[0]).astype(np.int)
    new_x = np.floor(coords[1] / grid[1]).astype(np.int)

    return new_y, new_x

def location_features(template, observation, grid, threshold=0.8):

    assert type(grid) == type(np.array([0])), type(grid)
    assert observation.shape[0] % grid[0] == 0
    assert observation.shape[1] % grid[1] == 0

    result = matchTemplate(observation, template, TM_CCORR_NORMED)
    loc = np.where(result > threshold)
    tiled = tile(loc, grid=grid)

    _new_shape = np.divide(observation.shape, grid).astype(np.int)
    _zeros = np.zeros(_new_shape)
    _zeros[tiled] = 1
    flat = _zeros.flatten()

    return flat

def extract_features(observation, templates, grid=np.array((5,5))):

    gray = cvtColor(observation, COLOR_BGR2GRAY)

    locs = []
    for k, v in templates.items():
        if 'enemy' not in k:
            continue

        _features = location_features(v, gray, grid=grid)
        locs.append(_features)

    _self = location_features(templates['self'], gray, grid=grid)
    locs.append(_self)

    locs = np.concatenate(locs)

    return locs

class HandcraftedDQN(nn.Module):

    def __init__(self, num_input, num_actions, device='cpu'):
        super(HandcraftedDQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(num_input, 300),
            nn.ReLU(),
            nn.Linear(300, num_actions)
        )

        def read_averaged_template(path_a, path_b):
            a = imread(path_a, 0)
            b = imread(path_b, 0)

            avg = addWeighted(a,0.5,b,0.5,0)

            return avg


        template_path = 'dqn/space_invader_sprites/enemy_{}_{}.png'

        self.templates = {
            'enemy_0' : read_averaged_template(template_path.format(0, 'a'), template_path.format(0, 'b')),
            'enemy_1' : read_averaged_template(template_path.format(1, 'a'), template_path.format(1, 'b')),
            'enemy_2' : read_averaged_template(template_path.format(2, 'a'), template_path.format(2, 'b')),
            'enemy_3' : read_averaged_template(template_path.format(3, 'a'), template_path.format(3, 'b')),
            'enemy_4' : read_averaged_template(template_path.format(4, 'a'), template_path.format(4, 'b')),
            'enemy_5' : read_averaged_template(template_path.format(5, 'a'), template_path.format(5, 'b')),
            'self' : imread('dqn/space_invader_sprites/my_sprite.png', 0),
            'defense' : imread('dqn/space_invader_sprites/defense.png', 0),
        }

        self.device = device
        self.to(device)

    def prepare_input(self, x):
        x = self._extract_features(x)
        x = torch.tensor(x, dtype=torch.float)
        x = x.to(self.device)
        return x

    def _extract_features(self, observation, grid=np.array([5,5])):
        return extract_features(observation, self.templates, grid)

    def forward(self, x):
        return self.net(x)