import random
import numpy as np
import torch
import torchvision
from collections import deque

def get_observation(env, action=None):
    if action is None:
        obs = env.reset()
        reward, done = None, None
    else:
        obs, reward, done, _ = env.step(action)

    return obs, reward, done

def transform_observation(obs, size):
    transformed_obs = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Resize(size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0], [1])
    ])(obs)

    return transformed_obs

def select_action(model, state, action_dims, epsilon):
    sample = random.random()
    if sample > epsilon:
        with torch.no_grad():
            # t.max(1) will return largest value for column of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return model(state.unsqueeze(0)).max(1)[1].view(1, 1)
    else:
        return random_action(action_dims, model.device)

def random_action(action_dims, device):
    return torch.tensor([[random.randrange(action_dims)]], device=device, dtype=torch.long)

def get_epsilon(it, start, stop, steps):
    it = min(it, steps-1)
    epsilons = np.linspace(start, stop, steps)

    return epsilons[it]

def generate_validation_states(env, model, depth, k):
    obs, _, _ = get_observation(env)
    obs = model.prepare_input(obs)

    history = deque(iterable=depth*[obs], maxlen=depth)

    candidate_states = [torch.cat(list(history), dim=0)]

    for _ in range(10*k):
        rand_action = random.randrange(env.action_space.n)
        obs, reward, done = get_observation(env, rand_action)
        obs = model.prepare_input(obs)

        history.append(obs)
        candidate_states.append(torch.cat(list(history), dim=0))

        if done:
            env.reset()

    sample = random.sample(candidate_states, k)
    sample = torch.stack(sample, dim=0)

    return sample