from boat_env import BoatEnvironment, Boat
import pygame as pg
import numpy as np
from vicero.policy import RandomPolicy, KeyboardPolicy

env = BoatEnvironment((960, 960))

pg.init()
screen = pg.display.set_mode(env.dimensions)
clock = pg.time.Clock()

framerate = 30

running = True

boat2_policy = RandomPolicy(env.action_space)
boat2_state = [0, 0]

boat1_policy = KeyboardPolicy(Boat.NOP, up=Boat.INC_SPD, down=Boat.DEC_SPD, left=Boat.TURN_CCW, right=Boat.TURN_CW)
boat1_state = [0, 0]

while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            running = False
        
    keys = pg.key.get_pressed() 
    
    boat1_state, _, _, _ = env.boat1.step(boat1_policy(keys))
    boat2_state, _, _, _ = env.boat2.step(boat2_policy(boat2_state))
    
    env.draw(screen)

    pg.display.flip()
    clock.tick(framerate)