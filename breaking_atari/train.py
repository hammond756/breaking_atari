import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

import time
import os
from collections import defaultdict
import pprint
import pandas as pd
import numpy as np

from breaking_atari.utils import select_action, get_epsilon, random_action, generate_validation_states
from breaking_atari.models.memory import ReplayBuffer

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


def gather_experience(model, env, queue, config):

    frames = 0
    action_dims = env.action_space.n

    rewards = []
    episodes = 0
    last_frame_at = time.time()

    while True:
        # Initialize the environment
        obs = env.reset()
        total_reward = 0
        done = False

        while not done:

            if frames % 100 == 0:
                current_time = time.time()
                hundred_frames_in = current_time - last_frame_at
                last_frame_at = current_time
                print("{} / {} frames done \t\t\t at {} f/s".format(frames, config.num_frames, 100/hundred_frames_in))

            # get epsilon based on number of frames
            epsilon = get_epsilon(frames, config.eps_start, config.eps_stop, config.eps_steps)

            if frames > config.exploration_phase:
                action = select_action(model, obs, action_dims, epsilon)
            else:
                action = random_action(action_dims, model.device)


            # place collected experience in the queue for the main process to consume
            next_obs, reward, done, _ = env.step(action.item())
            queue.put((obs, action.cpu().numpy(), reward, next_obs, done))

            # Move to the next state
            obs = next_obs

            frames += 1
            total_reward += reward

            if done:
                rewards.append(total_reward)
                episodes += 1

        if frames > config.num_frames:
            break

    queue.close()
    print("-- Done playing --")



def train(model, target, env, config):

    print('------')
    print('Starting {}'.format(config.environment), 'on', model.device)
    pprint.pprint(config)
    print('------')

    print('MODEL', model)

    optimizer = torch.optim.Adam(model.parameters(),lr=config.lr)
    replay_buffer = ReplayBuffer(config.memory)

    frames = 0
    queue = mp.Queue(maxsize=config.optimize_every * 2) # maxsize to keep processes somewhat in sync??
    play_process = mp.Process(target=gather_experience, args=(model, env, queue, config), name='Play')
    play_process.start()

    with torch.no_grad():
        model.eval()
        val_states = [env.reset()] # track expected value of initial state as proxy for learning
        # val_states = generate_validation_states(env, config.n_validation_states)

    while play_process.is_alive():
        for _ in range(config.optimize_every):
            frames += 1
            experience = queue.get()
            if experience is None:
                play_process.join() # wait for experience to accumulate
                break
            obs, action, reward, next_obs, done = experience
            replay_buffer.add(obs, action, reward, next_obs, done)

        # first acquire enough experience for samples to be decorrelated
        if len(replay_buffer) > config.exploration_phase:
            # Perform one step of the optimization (on the target network)
            model.train()
            optimize_model(model, target, replay_buffer, optimizer, config)

        # Update the target network, copying all weights and biases in DQN
        if frames % config.target_update == 0:
            target.load_state_dict(model.state_dict())

        if frames % config.eval_every == 0:
            model.eval()
            with torch.no_grad():
                avg_reward, max_reward, avg_duration = evaluate_reward(model, env, config)
                avg_q = evaluate_q_func(model, val_states)

            # update statistics
            # stats['epoch'].append(len(stats['epoch']))
            # stats['avg_reward'].append(avg_reward)
            # stats['max_reward'].append(max_reward)
            # stats['avg_duration'].append(avg_duration)
            # stats['avg_q'].append(avg_q)
            # stats['episodes'].append(episodes)

            # save model
            # model_path = config.output_dir + '/dqn-{}'.format(len(stats['epoch']))
            # torch.save(model.state_dict(), model_path)

            # save statistics
            # statistics = pd.DataFrame(stats)
            # statistics.to_csv(os.path.join(config.output_dir, 'stats.csv'), index_label='idx')


def evaluate_q_func(model, val_states):
    val_states = model.prepare_input(val_states, batch=True)
    predictions = model(val_states)
    max_q, argmax_q = torch.max(predictions, dim=1)

    avg_q = max_q.mean().item()

    print("Avg Q(s,a): \t\t{:.5}".format(avg_q))

    return avg_q

def evaluate_reward(model, env, config):
    action_dims = env.action_space.n

    rewards = []
    durations = []
    frames = 0

    while True:
        obs = env.reset()
        episode_reward = 0
        episode_duration = 0
        done = False

        while not done:
            # Select and perform an action
            epsilon = 0.05
            action = select_action(model, obs, action_dims, epsilon)
            next_obs, reward, done, _ = env.step(action.item())
            episode_reward += reward

            # Move to the next state
            obs = next_obs
            frames += 1
            episode_duration += 1

            if done:
                rewards.append(episode_reward)
                durations.append(episode_duration)

        if frames > config.num_eval:
            break

    avg_reward = sum(rewards) / len(rewards)
    max_reward = max(rewards)
    avg_duration = sum(durations) / len(durations)

    print('Evaluated on {} episodes'.format(len(rewards)))
    print('Avg reward: \t\t{}'.format(avg_reward))
    print('Max reward: \t\t{}'.format(avg_reward))
    print('Avg Duration:\t\t{}'.format(avg_duration))


    return avg_reward, max_reward, avg_duration
