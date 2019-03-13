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

env = BoatEnvironment(dim, screen)

framerate = 30

running = True

spec = NetworkSpecification(hidden_layer_sizes=[12, 8], activation_function=nn.Sigmoid)
dqn = DQN(env, spec, render=False, alpha=0.001, epsilon_end=0.1, memory_length=10000)

batch_size = 16
num_episodes = 200
training_iter = 240

print('training...')
dqn.train(num_episodes, batch_size, training_iter, verbose=True, plot=True, eps_decay=True)
boat1_policy = dqn.copy_target_policy(verbose=True)
boat2_policy = dqn.copy_target_policy(verbose=True)
dqn.save('alphahelge.pkl')

boat1_state = env.reset()
boat2_state = boat1_state

#boat1_policy([0.0, 0.0, 0.0, 0.0])

plt.plot(dqn.maxq_history)
plt.show()

while running:

    keys = pg.key.get_pressed()

    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            running = False
    
    if keys[pg.K_SPACE]: env.reset()

    boat1_state, _, done, _ = env.boat1.step(boat1_policy(boat1_state))
    boat2_state, _,    _, _ = env.boat2.step(boat2_policy(boat2_state))
    
    env.draw(screen)
    if done:
        env.reset()
    pg.display.flip()
    clock.tick(framerate)