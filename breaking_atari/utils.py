import random
import numpy as np
import torch
import torchvision
from collections import deque

def select_action(model, obs, action_dims, epsilon):
    sample = random.random()
    if sample > epsilon:
        with torch.no_grad():
            # t.max(1) will return largest value for column of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            state_tensor = model.prepare_input(obs, batch=False)
            return model(state_tensor).max(1)[1].view(1, 1)
    else:
        return random_action(action_dims, model.device)

def random_action(action_dims, device):
    return torch.tensor([[random.randrange(action_dims)]], device=device, dtype=torch.long)

def get_epsilon(it, start, stop, steps):
    it = min(it, steps-1)
    epsilons = np.linspace(start, stop, steps)

    return epsilons[it]

def generate_validation_states(env, k):
    obs = env.reset()

    candidate_states = [obs]
    for _ in range(10*k):
        rand_action = random.randrange(env.action_space.n)
        obs, reward, done, _ = env.step(rand_action)
        candidate_states.append(obs)

        if done:
            env.reset()

    sample = random.sample(candidate_states, k)

    return sample
