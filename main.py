from boat_env import BoatEnvironment, Boat
import pygame as pg
import numpy as np
from vicero.policy import RandomPolicy, KeyboardPolicy
from vicero.algorithms.deepqlearning import DQN
from vicero.algorithms.common.neuralnetwork import NetworkSpecification

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

dim = (400, 400)

pg.init()
screen = pg.display.set_mode(dim)
clock = pg.time.Clock()

env = BoatEnvironment(dim, screen, intermediate_rewards=False, multi_agent=True, default_reward=0)

framerate = 60
running = True

spec = NetworkSpecification(hidden_layer_sizes=[14, 8], activation_function=nn.ReLU)
dqn = DQN(env, spec, render=True, gamma=0.99, alpha=1e-2, epsilon_start=0.8, epsilon_end=0.1, memory_length=5000)

batch_size = 4
num_episodes = 200 
training_iter = 640

print('training...')
dqn.train(num_episodes, batch_size, training_iter, verbose=True, plot=True, eps_decay=True)
boat1_policy = dqn.copy_target_policy(verbose=False)
boat2_policy = dqn.copy_target_policy(verbose=False)
dqn.save('baat.pkl')

boat1_state = env.reset()
boat2_state = boat1_state

plt.plot(dqn.history)
plt.show()

plt.plot(dqn.loss_history)
plt.show()

plt.plot(dqn.maxq_history)
plt.show()

plt.plot(dqn.history)
plt.plot(dqn.maxq_history)
plt.show()

while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            running = False

    keys = pg.key.get_pressed()    
    
    if keys[pg.K_SPACE]: env.reset(randomize=False)

    sm1 = dqn.action_distribution(boat1_state)
    sm2 = dqn.action_distribution(boat2_state)

    boat1_state, _, done1, _ = env.boat1.step(boat1_policy(boat1_state))
    boat2_state, _, done2, _ = env.boat2.step(boat2_policy(boat2_state))
    
    env.draw(screen)

    bar_w = dim[0] // 18
    for i in range(len(sm1)):
        pg.draw.rect(screen, (0, 200, 0), pg.Rect(0 + i * bar_w, dim[1], bar_w, -(dim[1] // 2) * sm1[i]))
        pg.draw.rect(screen, (0, 160, 0), pg.Rect(0 + i * bar_w, dim[1], bar_w, -(dim[1] // 2) * sm1[i]), 1)
        pg.draw.rect(screen, (200, 200, 0), pg.Rect(dim[0] // 2 + i * bar_w, dim[1], bar_w, -(dim[1] // 2) * sm2[i]))
        pg.draw.rect(screen, (160, 160, 0), pg.Rect(dim[0] // 2 + i * bar_w, dim[1], bar_w, -(dim[1] // 2) * sm2[i]), 1)
    
    if done1 or done2: env.reset()

    pg.display.flip()
    clock.tick(framerate)