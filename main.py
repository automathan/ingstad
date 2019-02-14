from boat_env import BoatEnvironment, Boat
import pygame as pg

env = BoatEnvironment((960, 960))

pg.init()
screen = pg.display.set_mode(env.dimensions)
clock = pg.time.Clock()

framerate = 30

running = True
while running:
    action = Boat.NOP
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            running = False
        
    keys = pg.key.get_pressed() 
    if keys[pg.K_UP]:    action = Boat.INC_SPD
    if keys[pg.K_DOWN]:  action = Boat.DEC_SPD
    if keys[pg.K_LEFT]:  action = Boat.TURN_CCW
    if keys[pg.K_RIGHT]: action = Boat.TURN_CW

    action2 = Boat.NOP
    if keys[pg.K_w]: action2 = Boat.INC_SPD
    if keys[pg.K_s]: action2 = Boat.DEC_SPD
    if keys[pg.K_a]: action2 = Boat.TURN_CCW
    if keys[pg.K_d]: action2 = Boat.TURN_CW

    env.boat1.step(action)
    env.boat2.step(action2)
    env.draw(screen)

    pg.display.flip()
    clock.tick(framerate)