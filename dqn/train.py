import torch
import torch.nn.functional as F

import gym
import os
from collections import deque, defaultdict
import pprint
import pandas as pd
import numpy as np


from model.utils import select_action, get_epsilon, random_action, generate_validation_states
from model.memory import ReplayBuffer

def optimize_model(model, target, memory, optimizer, config):
    # if len(memory) < config.batch_size:
    #     return
    states, actions, rewards, next_states, dones = memory.sample(config.batch_size)

    states = model.prepare_input(states, batch=True)
    next_states = model.prepare_input(next_states, batch=True)

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.as_tensor(np.logical_not(dones).astype(np.uint8), device=model.device)
    non_final_next_states = next_states[non_final_mask]

    action_batch = torch.as_tensor(np.concatenate(actions), device=model.device, dtype=torch.long)
    reward_batch = torch.as_tensor(np.concatenate(rewards[None,:]), device=model.device, dtype=torch.float)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = model(states).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(config.batch_size, device=model.device)
    target_output = target(non_final_next_states).max(1)[0].detach()
    next_state_values[non_final_mask] = target_output

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

def train(model, target, env, config):
    action_dims = env.action_space.n
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('------')
    print('Starting {}'.format(config.environment), 'on', device)
    pprint.pprint(config)
    print('------')

    optimizer = torch.optim.Adam(model.parameters(),lr=config.lr)
    replay_buffer = ReplayBuffer(config.memory)

    frames = 0
    episodes = 0
    rewards = []
    stats = defaultdict(list)

    with torch.no_grad():
        model.eval()
        val_states = generate_validation_states(env, model.device, 512)

    while True:
        # Initialize the environment
        obs = env.reset()
        total_reward = 0

        while True:

            if frames % 100 == 0:
                print("{} / {} frames done".format(frames, config.num_frames))

            # get epsilon based on number of frames
            epsilon = get_epsilon(frames, config.eps_start, config.eps_stop, config.eps_steps)

            # first acquire enough experience for samples to be decorrelated
            if frames > config.exploration_phase:
                action = select_action(model, obs, action_dims, epsilon)
            else:
                action = random_action(action_dims, model.device)

            next_obs, reward, done, _ = env.step(action.item())

            frames += 1
            total_reward += reward

            replay_buffer.add(obs, action, reward, next_obs, done)

            # Move to the next state
            obs = next_obs

            if frames % 4 == 0 and len(replay_buffer) > config.exploration_phase:
                # Perform one step of the optimization (on the target network)
                model.train()
                optimize_model(model, target, replay_buffer, optimizer, config)

            # Update the target network, copying all weights and biases in DQN
            if frames % config.target_update == 0:
                target.load_state_dict(model.state_dict())

            if frames % config.eval_every == 0:
                model.eval()
                with torch.no_grad():
                    avg_reward, max_reward = evaluate_reward(model, env, config)
                    avg_q = evaluate_q_func(model, val_states)

                print('Average reward: \t\t{}'.format(avg_reward))
                print('Average q-val: \t\t{}'.format(avg_q))

                # update statistics
                stats['epoch'].append(len(stats['epoch']))
                stats['avg_reward'].append(avg_reward)
                stats['max_reward'].append(max_reward)
                stats['avg_q'].append(avg_q)
                stats['episodes'].append(episodes)

                # save model
                model_path = config.output_dir + '/dqn-{}'.format(len(stats['epoch']))
                torch.save(model.state_dict(), model_path)

                # save statistics
                statistics = pd.DataFrame(stats)
                statistics.to_csv(os.path.join(config.output_dir, 'stats.csv'), index_label='idx')

            if done:
                rewards.append(total_reward)
                episodes += 1
                break

        if frames > config.num_frames:
            break

def evaluate_q_func(model, val_states):
    predictions = model(val_states)
    max_q, argmax_q = torch.max(predictions, dim=1)

    return max_q.mean().item()

def evaluate_reward(model, env, config):
    action_dims = env.action_space.n

    rewards = []
    frames = 0

    while True:
        obs = env.reset()
        history = deque(iterable=config.frame_stack*[obs], maxlen=config.frame_stack)
        state = np.concatenate(list(history), axis=0)

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

    avg_reward = sum(rewards) / len(rewards)
    max_reward = max(rewards)

    return avg_reward, max_reward
