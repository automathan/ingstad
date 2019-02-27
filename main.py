from boat_env import BoatEnvironment, Boat
import pygame as pg
import numpy as np
from vicero.policy import RandomPolicy, KeyboardPolicy
from vicero.algorithms.deepqlearning import DQN
from vicero.algorithms.common.neuralnetwork import NetworkSpecification

import torch
import torch.nn as nn


dim = (400, 400)

pg.init()
screen = pg.display.set_mode(dim)
clock = pg.time.Clock()

env = BoatEnvironment(dim, screen)

framerate = 30

running = True

boat1_policy = RandomPolicy(env.action_space)
boat1_state = [0, 0]

boat2_policy = KeyboardPolicy(Boat.NOP, up=Boat.INC_SPD, down=Boat.DEC_SPD, left=Boat.TURN_CCW, right=Boat.TURN_CW)
boat2_state = [0, 0]

spec = NetworkSpecification(hidden_layer_sizes=[16], activation_function=nn.Sigmoid)
dqn = DQN(env, spec, render=False, epsilon=0.4, alpha=0.01, memory_length=500)

batch_size = 16
num_episodes = 5000
training_iter = 50
completion_reward = -10

print('training...')
dqn.train(num_episodes, batch_size, training_iter, verbose=True, completion_reward=completion_reward, plot=True, eps_decay=False)
boat1_policy = dqn.copy_target_policy()#verbose=True)

state = env.reset()
while running:

    keys = pg.key.get_pressed() 

    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            running = False
    
    if keys[pg.K_SPACE]: env.reset()

    state, _, done, _ = env.boat1.step(boat1_policy(state))
    
    boat2_state, _, _, _ = env.boat2.step(boat2_policy(keys))
    
    env.draw(screen)
    if done:
        env.reset()
    pg.display.flip()
    clock.tick(framerate)