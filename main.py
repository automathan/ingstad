from boat_env import BoatEnvironment, Boat
import pygame as pg
import numpy as np
from vicero.policy import RandomPolicy, KeyboardPolicy
from vicero.algorithms.deepqlearning import DQN
from vicero.algorithms.common.neuralnetwork import NetworkSpecification
from vicero.visualization.overlay import ActionDistributionOverlay as ADO
import pandas as pd

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

dim = (480, 480)

pg.init()
screen = pg.display.set_mode(dim)
clock = pg.time.Clock()

env = BoatEnvironment(dim, screen, 
    intermediate_rewards=False,
    multi_agent=True,
    default_reward=0,
    illegal_penalty=-1
)

framerate = 60
running = True

spec = NetworkSpecification(hidden_layer_sizes=[14, 8], activation_function=nn.ReLU)
dqn = DQN(env, spec, render=False, gamma=0.99, alpha=1e-5, epsilon_start=0.9, epsilon_end=0.05, memory_length=5000)

batch_size = 2
num_episodes = 20
training_iter = 10

print('training...')
dqn.train(num_episodes, batch_size, training_iter, verbose=True, plot=True, eps_decay=True)
boat1_policy = dqn.copy_target_policy(verbose=False)
boat2_policy = dqn.copy_target_policy(verbose=False)
dqn.save('long_training.pkl')

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

ado = ADO(dqn, pg.Rect(0, 0, 100, 100))

while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            running = False

    keys = pg.key.get_pressed()    
    
    if keys[pg.K_SPACE]: env.reset(randomize=False)

    boat1_state, _, done1, _ = env.boat1.step(boat1_policy(boat1_state))
    boat2_state, _, done2, _ = env.boat2.step(boat2_policy(boat2_state))
    
    env.draw(screen)
    ado.render(screen, boat1_state)
    
    if done1 or done2: env.reset()

    pg.display.flip()
    clock.tick(framerate)