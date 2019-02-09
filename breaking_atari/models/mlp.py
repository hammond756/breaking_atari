import torch
nn = torch.nn
tensor = torch.tensor

class MLP(nn.Module):

    def __init__(self, num_input, num_hidden, num_actions, device):
        super(MLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(num_input, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_actions)
        )

        self.device = device
        self.to(device)

    def _prepare_single_input(self, x):
        x = tensor(x, dtype=torch.float, device=self.device).unsqueeze(0)
        return x

    def prepare_input(self, x, batch=False):
        if not batch:
            return self._prepare_single_input(x)
        else:
            return torch.cat([self._prepare_single_input(_x) for _x in x])

    def forward(self, x):
        return self.net(x)
