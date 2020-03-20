import copy
import random
import time
from pathlib import Path

import gym
import torch
from torch.nn import MSELoss
from torch.optim import RMSprop
from torch.utils.data import DataLoader

from memory import Memory
from neural_net import CnnNeuralNet
from utils import make_gif


def train_step(model, eval_net, data_loader, gamma, loss_fn, optimizer):
    for i_batch, batch in enumerate(data_loader):
        with torch.no_grad():
            next_value = torch.max(eval_net(batch['next_state']), dim=1)[0]

        actual_value = torch.where(
            batch['done'],
            batch['reward'],
            batch['reward'] + gamma * next_value).unsqueeze(-1)

        my_value = model(batch['state']).gather(1, batch['action'].unsqueeze(-1))
        optimizer.zero_grad()
        err = loss_fn(my_value, actual_value)
        err.backward()
        optimizer.step()


def play_episode(model, env, memory, save_path):
    memory.clear()
    frame = env.reset()
    done = False
    cumulative_reward = 0
    frames = []
    while not done:
        frames.append(frame)
        with torch.no_grad():
            action = model(memory.get_state(frame)).max(1)[1][0].cpu()  # .numpy()

        next_frame, reward, done, info = env.step(action)
        cumulative_reward += reward
        memory.add_experience(frame, action, reward, next_frame, done)

        frame = next_frame

    save_path = Path(save_path.parent, save_path.stem + '_r' + str(int(cumulative_reward)) + save_path.suffix)
    # save_path = save_path.name + '_r' + cumulative_reward + save_path.suffix
    make_gif(frames, save_path)
    return cumulative_reward


def train_whole(env, memory, epsilon_start,
                epsilon_end,
                epsilon_reach_end_steps, model,
                eval_net, gamma,
                loss_fn, optimizer,
                num_burn_in,
                train_freq,
                num_iterations,
                target_update_freq,
                play_freq,
                batch_size,
                gif_save_root,
                model_save_freq,
                model_save_root,
                stats_show_period=100):
    env_copy = copy.deepcopy(env)
    memory_copy = copy.deepcopy(memory)

    frame = env.reset()
    cumulative_reward = 0
    start_time = time.time()

    eps_step = (epsilon_start - epsilon_end) / epsilon_reach_end_steps
    print(eps_step)
    for idx in range(1, num_iterations + 1):
        if idx < num_burn_in or random.uniform(0, 1) < epsilon_start:
            action = torch.tensor(env.action_space.sample(), dtype=torch.long)
        else:
            with torch.no_grad():
                action = model(memory.get_state(frame)).max(1)[1][0].cpu()  # .numpy()

        next_frame, reward, done, info = env.step(action)
        memory.add_experience(frame, action, reward, next_frame, done)
        frame = next_frame
        cumulative_reward += reward
        if done:
            frame = env.reset()
            print("cumulative reward", cumulative_reward)
            cumulative_reward = 0

        if idx > num_burn_in and len(memory) > batch_size and idx % train_freq == 0:
            train_step(model, eval_net,
                       DataLoader(memory, batch_size=batch_size, shuffle=True),
                       gamma, loss_fn, optimizer)

        if idx % target_update_freq == 0:
            eval_net.load_state_dict(model.state_dict())

        if idx % play_freq == 0 and idx > num_burn_in:
            play_episode(model, env_copy, memory_copy, Path(gif_save_root, 'idx_e{}.gif'.format(idx)))

        if idx % stats_show_period == 0:
            print("iter: {}, cumulative reward: {}, time: {}s".format(idx, cumulative_reward,
                                                                      time.time() - start_time))
            start_time = time.time()

        if idx % model_save_freq == 0 and idx > num_burn_in:
            my_save_root = Path(model_save_root, 'idx_' + str(idx))
            my_save_root.mkdir(exist_ok=True, parents=True)
            torch.save(model.state_dict(), Path(my_save_root, 'model.pth'))
            torch.save(eval_net.state_dict(), Path(my_save_root, 'eval_model.pth'))

        if idx > num_burn_in:
            epsilon_start -= eps_step


def main():
    memory_size = int(1e5)
    history_images = 3
    batch_size = 64
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_reach_end_steps = 5000
    gamma = 0.95
    lr = 15e-4
    train_freq = 3
    target_update_freq = 100
    num_burn_in = 3e3  # int(5e4)
    num_iterations = int(3e4)
    play_freq = 500
    model_save_freq = 400
    frame_shape = (128, 84)

    gif_save_root = Path('gifs')
    gif_save_root.mkdir(exist_ok=True, parents=True)

    model_save_root = Path('models')
    model_save_root.mkdir(exist_ok=True, parents=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    env = gym.make("SpaceInvaders-v0")
    # print(env.observation_space.shape)
    # frame_shape = env.observation_space.shape

    memory = Memory(memory_size, history_images, frame_shape, device=device)
    model = CnnNeuralNet((history_images, frame_shape[0], frame_shape[1]), env.action_space.n)
    model = model.to(device)
    # if device.type == 'cuda':
    #     model.half()
    eval_net = copy.deepcopy(model)
    loss_fn = MSELoss()
    optimizer = RMSprop(params=model.parameters(), lr=lr)  # Adam(params=model.parameters(), lr=lr)

    train_whole(
        env,
        memory,
        epsilon_start,
        epsilon_end,
        epsilon_reach_end_steps,
        model,
        eval_net,
        gamma,
        loss_fn,
        optimizer,
        num_burn_in,
        train_freq,
        num_iterations,
        target_update_freq,
        play_freq,
        batch_size,
        gif_save_root,
        model_save_freq,
        model_save_root
    )


if __name__ == '__main__':
    main()
