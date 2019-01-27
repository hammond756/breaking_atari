import random
import numpy as np
import torch
import torchvision

def get_observation(env, action=None):
    if action is None:
        obs = env.reset()
        reward, done = None, None
    else:
        obs, reward, done, _ = env.step(action)

    return obs, reward, done

def transform_observation(obs, size):
    return torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Resize(size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0], [1])
    ])(obs)

def select_action(model, state, action_dims, epsilon):
    sample = random.random()
    if sample > epsilon:
        with torch.no_grad():
            # t.max(1) will return largest value for column of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return model(state.unsqueeze(0)).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(action_dims)]], device=model.device, dtype=torch.long)

def get_epsilon(it, start, stop, steps):
    it = min(it, steps-1)
    epsilons = np.linspace(start, stop, steps)

    return epsilons[it]