import torch
import torch.nn.functional as F

import gym
import os
from collections import deque
import pprint
import pandas as pd


from model.utils import select_action, get_observation, transform_observation, get_epsilon, random_action, generate_validation_states
from model.memory import Transition
from model.dqn import HandcraftedDQN, DQN, extract_features
from model.memory import ReplayMemory

def optimize_model(model, target, memory, optimizer, config):

    transitions = memory.sample(config.batch_size)
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
    next_state_values = torch.zeros(config.batch_size, device=model.device)
    next_state_values[non_final_mask] = target(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * config.gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def train(config):
    env = gym.make(config.environment)
    action_dims = env.action_space.n
    depth = config.frame_stack
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('------')
    print('Starting {}'.format(config.environment), 'on', device)
    pprint.pprint(config)
    print('------')

    if not os.path.isdir(config.output_dir):
        os.mkdir(config.output_dir)

    height, width = config.image_size
    model = DQN(height, width, action_dims, device)

    target = DQN(height, width, action_dims, device)
    target.load_state_dict(model.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(model.parameters(),lr=config.lr)
    replay_buffer = ReplayMemory(config.memory)

    frames = 0
    episodes = 0

    rewards = []

    stats = {
        'epoch' : [],
        'avg_reward' : [],
        'avg_q' : [],
        'max_reward' : [],
        'episodes' : []
    }

    val_states = generate_validation_states(env, model, depth, 32)

    while True:
        # Initialize the environment and state
        obs, _, _ = get_observation(env)
        obs = model.prepare_input(obs)
        history = deque(iterable=depth*[obs], maxlen=depth)

        state = torch.cat(list(history), dim=0)

        total_reward = 0

        while True:

            if frames % 100 == 0:
                print("{} / {} frames done".format(frames, config.num_frames))

            # Select and perform an action
            epsilon = get_epsilon(frames, config.eps_start, config.eps_stop, config.eps_steps)

            # first acquire enough experience for samples to be decorrelated
            if frames > config.exploration_phase:
                action = select_action(model, state, action_dims, epsilon)
            else:
                action = random_action(action_dims, model.device)

            next_obs, reward, done = get_observation(env, action.item())
            next_obs = model.prepare_input(next_obs)
            history.append(next_obs)
            next_state = torch.cat(list(history), dim=0)

            frames += 1

            total_reward += reward
            reward = torch.tensor([reward], device=model.device)

            # Store the transition in memory
            replay_buffer.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            if frames % 4 == 0 and len(replay_buffer) > config.exploration_phase:
                # Perform one step of the optimization (on the target network)
                model.train()
                optimize_model(model, target, replay_buffer, optimizer, config)

            # Update the target network, copying all weights and biases in DQN
            if frames % config.target_update == 0:
                target.load_state_dict(model.state_dict())

            if frames % config.eval_every == 0:
                model.eval()
                avg_reward = evaluate_reward(model, env, config)
                avg_q = evaluate_q_func(model, val_states)

                print('Average reward: \t\t{}'.format(avg_reward))
                print('Average q-val: \t\t{}'.format(avg_q))

                stats['epoch'].append(len(stats['epoch']))
                stats['avg_reward'].append(avg_reward)
                stats['avg_q'].append(avg_q)
                stats['episodes'].append(episodes)

            if done:
                rewards.append(total_reward)
                episodes += 1


    statistics = pd.DataFrame(stats)
    statistics.to_csv(os.path.join(config.output_dir, 'stats.csv'), index_label='idx')



def evaluate_q_func(model, val_states):
    predictions = model(val_states)
    max_q, argmax_q = torch.max(predictions, dim=1)

    return max_q.mean().item()

def evaluate_reward(model, env, config):
    action_dims = env.action_space.n
    model.eval()

    rewards = []
    frames = 0

    while True:
        obs, _, _ = get_observation(env)
        obs = model.prepare_input(obs)
        history = deque(iterable=config.frame_stack*[obs], maxlen=config.frame_stack)
        state = torch.cat(list(history), dim=0)

        episode_reward = 0

        while True:
            # Select and perform an action
            epsilon = 0.05
            action = select_action(model, state, action_dims, epsilon)
            next_obs, reward, done = get_observation(env, action.item())
            next_obs = model.prepare_input(next_obs)
            history.append(next_obs)
            next_state = torch.cat(list(history), dim=0)
            episode_reward += reward

            # Move to the next state
            state = next_state

            frames += 1

            if done:
                rewards.append(episode_reward)
                break

        if frames > config.num_eval:
            break

    return sum(rewards) / len(rewards)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # defaults come from Mnih et al. (2015)
    parser.add_argument('--batch_size', type=str, required=False, default=32)
    parser.add_argument('-gamma', type=float, required=False, default=0.99)
    parser.add_argument('--eps_start', type=float, required=False, default=1.0)
    parser.add_argument('--eps_stop', type=float, required=False, default=0.1)
    parser.add_argument('--eps_steps', type=int, required=False, default=1000000)
    parser.add_argument('--image_size', type=int, nargs=2, required=False, default=[110, 84])
    parser.add_argument('--environment', type=str, required=True)
    parser.add_argument('--target_update', type=int, required=False, default=10000)
    parser.add_argument('--num_frames', type=int, required=False, default=10000000)
    parser.add_argument('--num_eval', type=int, required=False, default=10000)
    parser.add_argument('--eval_every', type=int, required=False, default=100000)
    parser.add_argument('--lr', type=float, required=False, default=0.00025)
    parser.add_argument('--memory', type=int, required=False, default=1000000)
    parser.add_argument('--exploration_phase', type=int, required=False, default=50000)
    parser.add_argument('--frame_stack', type=int, required=False, default=4)
    parser.add_argument('--output_dir', type=str, required=True)

    config = parser.parse_args()
    train(config)
