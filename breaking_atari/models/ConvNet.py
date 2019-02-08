__author__ = 'Aron'

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class ConvNet(nn.Module):

    def __init__(self, c, h, w, action_dims, device='cpu'):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(c, 16, kernel_size=5, stride=2)
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

    def _prepare_single_input(self, x):
        return torch.as_tensor(np.stack(x)[None, :, :, :].transpose((0,3,1,2)), dtype=torch.float, device=self.device)

    def prepare_input(self, x, batch=False):
        if not batch:
            return self._prepare_single_input(x)
        elif batch:
            return torch.cat([self._prepare_single_input(_x) for _x in x])

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.fc(x.view(x.size(0), -1))