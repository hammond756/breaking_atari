import torch
import torch.nn.functional as F

import gym
from itertools import count

from model.utils import select_action, get_observation, transform_observation
from model.memory import Transition
from model.dqn import HandcraftedDQN, DQN, extract_features
from model.memory import ReplayMemory
from constants import BATCH_SIZE, GAMMA, TARGET_UPDATE, IMAGE_DIMS

def optimize_model(model, target, memory, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                            device=model.device, dtype=torch.uint8)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=model.device)
    next_state_values[non_final_mask] = target(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        # param.grad.data.clamp_(-1, 1)
        pass
    optimizer.step()

env = gym.make('SpaceInvaders-v0')
# get properties from environment
height, width = IMAGE_DIMS
action_dims = env.action_space.n

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# initialze models
model = HandcraftedDQN(9408, action_dims).to(device)
target = HandcraftedDQN(9408, action_dims).to(device)
target.load_state_dict(model.state_dict())
target.eval()

optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
replay_buffer = ReplayMemory(100000)

steps_done = 0

num_episodes = 50
rewards = []

def train():
    for i_episode in range(num_episodes):
        print("starting {} / {}".format(i_episode, num_episodes))
        # Initialize the environment and state
        obs, _, _ = get_observation(env)

        import png
        png.from_array(obs, mode='RGB').save('initial.png')

        obs = extract_features(obs)
        obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.to(model.device)

        total_reward = 0

        for t in count():

            # Select and perform an action
            action = select_action(model, obs, steps_done, action_dims)
            next_obs, reward, done = get_observation(env, action.item())
            next_obs = extract_features(next_obs)
            next_obs = torch.tensor(next_obs, dtype=torch.float)
            next_obs = next_obs.to(device)

            total_reward += reward

            reward = torch.tensor([reward], device=model.device)

            if t % 300 == 0:
                print('{} seconds of game play'.format(t / 30))

            # Store the transition in memory
            replay_buffer.push(obs, action, next_obs, reward)

            # Move to the next state
            obs = next_obs

            # Perform one step of the optimization (on the target network)
            optimize_model(model, target, replay_buffer, optimizer)
            if done:
                rewards.append(total_reward)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target.load_state_dict(model.state_dict())

        print('reward for episode', total_reward)

    print('Complete')
    print(rewards)

if __name__ == '__main__':
    train()
