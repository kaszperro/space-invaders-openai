import copy
import random

import gym
import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from memory import Memory
from neural_net import CnnNeuralNet


def train_step(model, eval_net, data_loader, gamma, loss_fn, optimizer):
    for i_batch, batch in enumerate(data_loader):
        with torch.no_grad():
            next_value, _ = torch.max(eval_net(batch['next_state']), dim=1)

        actual_value = torch.where(batch['done'], batch['reward'], batch['reward'] + gamma * next_value)
        my_value = model(batch['state']).take(batch['action'])
        optimizer.zero_grad()
        err = loss_fn(my_value, actual_value)
        err.backward()
        optimizer.step()
        print(err)


def train_epoch(env, memory, epsilon, model, eval_net, gamma, loss_fn, optimizer, batch_size=64):
    frame = env.reset()

    it = 0
    done = False
    cumulative_reward = 0
    while not done:
        if len(memory) >= batch_size and it % 20 == 0:
            train_step(model, eval_net, DataLoader(memory, batch_size=batch_size, shuffle=True), gamma, loss_fn,
                       optimizer)

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = np.argmax(model(memory.get_state(frame, True)).numpy())

        next_frame, reward, done, info = env.step(action)
        cumulative_reward += reward

        if done:
            env.reset()

        memory.add_experience(frame, action, reward, next_frame, done)
        frame = next_frame
        it += 1

        if it % 50 == 0:
            eval_net.load_state_dict(model.state_dict())
            print(cumulative_reward)

    return cumulative_reward


def main():
    memory_size = 1000
    history_images = 2
    batch_size = 64
    epochs = 10
    epsilon = 0.1
    gamma = 0.8
    lr = 0.001

    env = gym.make("SpaceInvaders-v0")
    frame_shape = env.observation_space.shape

    memory = Memory(memory_size, history_images, (frame_shape[0], frame_shape[1]))
    model = CnnNeuralNet((history_images, frame_shape[0], frame_shape[1]), env.action_space.n, 1)
    eval_net = copy.deepcopy(model)
    loss_fn = MSELoss()
    optimizer = Adam(params=model.parameters(), lr=lr)
    for epoch in range(epochs):
        reward = train_epoch(env,
                             memory,
                             epsilon,
                             model,
                             eval_net=eval_net,
                             gamma=gamma,
                             loss_fn=loss_fn,
                             optimizer=optimizer,
                             batch_size=batch_size)

        print("e: {}".format(reward))


if __name__ == '__main__':
    main()
