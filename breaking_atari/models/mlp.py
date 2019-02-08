from cv2 import matchTemplate, TM_CCORR_NORMED, COLOR_BGR2GRAY, cvtColor, imread, addWeighted
import numpy as np
import torch
nn = torch.nn
tensor = torch.tensor

from breaking_atari.atari_wrappers.utils import extract_features

class MLP(nn.Module):

    def __init__(self, num_input, num_actions, sprites_dir, device='cpu'):
        super(MLP, self).__init__()

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


        template_path = sprites_dir + 'enemy_{}_{}.png'

        self.templates = {
            'enemy_0' : read_averaged_template(template_path.format(0, 'a'), template_path.format(0, 'b')),
            'enemy_1' : read_averaged_template(template_path.format(1, 'a'), template_path.format(1, 'b')),
            'enemy_2' : read_averaged_template(template_path.format(2, 'a'), template_path.format(2, 'b')),
            'enemy_3' : read_averaged_template(template_path.format(3, 'a'), template_path.format(3, 'b')),
            'enemy_4' : read_averaged_template(template_path.format(4, 'a'), template_path.format(4, 'b')),
            'enemy_5' : read_averaged_template(template_path.format(5, 'a'), template_path.format(5, 'b')),
            'self' : imread(sprites_dir + 'my_sprite.png', 0),
            'defense' : imread(sprites_dir + 'defense.png', 0),
        }

        self.device = device
        self.to(device)

    def _prepare_single_input(self, x):
        x = extract_features(x.squeeze(), self.templates, np.array([5,5]), 0.95)
        x = tensor(x, dtype=torch.float, device=self.device).unsqueeze(0)
        return x

    def prepare_input(self, x, batch=False):
        if not batch:
            return self._prepare_single_input(x)
        else:
            return torch.cat([self._prepare_single_input(_x) for _x in x])

    def forward(self, x):
        return self.net(x)
