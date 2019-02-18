from boat_env import BoatEnvironment, Boat
import pygame as pg
import numpy as np
from vicero.policy import RandomPolicy, KeyboardPolicy
from vicero.algorithms.deepqlearning import DQN
from vicero.algorithms.common.neuralnetwork import NetworkSpecification


dim = (960, 960)

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

spec = NetworkSpecification()
dqn = DQN(env, spec, render=True)

batch_size = 32
num_episodes = 5
training_iter = 5000
completion_reward = 5

print('training...')
dqn.train(num_episodes, batch_size, training_iter, verbose=True, completion_reward=completion_reward, plot=True, eps_decay=True)
boat1_policy = dqn.copy_target_policy()

state = env.reset()
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            running = False
        
    keys = pg.key.get_pressed() 
    
    state, _, _, _ = env.boat1.step(boat1_policy(state))
    
    boat2_state, _, _, _ = env.boat2.step(boat2_policy(keys))
    
    env.draw(screen)

    pg.display.flip()
    clock.tick(framerate)