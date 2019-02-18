import pygame as pg
import numpy as np
import pygame.freetype as pgft
import math

pg.init()
myimage = pg.image.load("boat.png")
imagerect = myimage.get_rect()
font = pgft.SysFont('Comic Sans MS', 28)
excl_mark, _ = font.render('!!!', (255, 0, 0))

class Boat:
    NOP, TURN_CW, TURN_CCW, INC_SPD, DEC_SPD = range(5)
    def __init__(self, pos, env, initial_speed=0, initial_acceleration=0, initial_direction=0, initial_angular_velocity=0, wrap=True):
        self.x = pos[0]
        self.y = pos[1]
        self.length = 96
        self.speed = initial_speed
        self.acceleration = initial_acceleration
        self.direction = initial_direction
        self.angular_velocity = initial_angular_velocity
        self.env = env
        self.other_ship_dist = -1
        self.max_ship_spd = 10
        self.min_ship_spd = -5
        self.show_circles = False
        self.goal_position = (0, 0)
        self.state = [0.0, 0.0, 0.0, 0.0]
        self.wrap = wrap

    def step(self, action):
        done = False

        if action == Boat.INC_SPD:  self.speed += 0.1
        if action == Boat.DEC_SPD:  self.speed -= 0.1
        if action == Boat.TURN_CCW: self.angular_velocity += 0.1
        if action == Boat.TURN_CW:  self.angular_velocity -= 0.1

        self.x += self.speed * math.cos(math.radians(self.direction))
        self.y += self.speed * -math.sin(math.radians(self.direction))
        
        if self.wrap:
            if self.x < 0: self.x = self.env.dimensions[0]
            if self.y < 0: self.y = self.env.dimensions[1]
            if self.x > self.env.dimensions[0]: self.x = 0
            if self.y > self.env.dimensions[1]: self.y = 0
        else:
            if self.x < 0: done = True
            if self.y < 0: done = True
            if self.x > self.env.dimensions[0]: done = True
            if self.y > self.env.dimensions[1]: done = True
        
        self.speed += self.acceleration
        self.direction += self.angular_velocity
        
        self.other_ship_dist = math.sqrt((self.env.boat1.x - self.env.boat2.x) ** 2 + (self.env.boat1.y - self.env.boat2.y) ** 2)
        dist_to_goal = math.sqrt((self.x - self.goal_position[0]) ** 2 + (self.y - self.goal_position[1]) ** 2)
        
        self.state = [self.x / self.env.dimensions[0], self.y / self.env.dimensions[1], 0.0, 0.0]
        reward = -1
        
        if dist_to_goal < 200:
            reward = 10

        return self.state, reward, done, {}

    def draw(self, screen):
        boat_image = pg.transform.rotate(myimage, self.direction)
        b_img_w, b_img_h = (boat_image.get_width(), boat_image.get_height())
        boat_pivot = (self.x - b_img_w / 2, self.y - b_img_h / 2)
        screen.blit(boat_image, imagerect.move(boat_pivot))
        
        if self.show_circles:
            pg.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), 4)
            pg.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), self.length // 2, 1)
            pg.draw.circle(screen, (255, 140, 0), (int(self.x), int(self.y)), self.length, 1)
            pg.draw.circle(screen, (255, 187, 0), (int(self.x), int(self.y)), self.length * 2, 1)
            pg.draw.circle(screen, (255, 250, 0), (int(self.x), int(self.y)), self.length * 4, 1)
        
        # line to other ship
        pg.draw.line(screen, (255, 0, 0), (self.env.boat1.x, self.env.boat1.y), (self.env.boat2.x, self.env.boat2.y))
        
        # forward line
        pg.draw.line(screen, (255, 0, 0), (self.x, self.y), (self.x + self.length * math.cos(math.radians(self.direction)), self.y + self.length * -math.sin(math.radians(self.direction))))
        
        # line to goal position
        pg.draw.line(screen, (255, 0, 0), (self.x, self.y), self.goal_position)
        
        if self.other_ship_dist < self.length * 4:
            screen.blit(excl_mark, (self.x - 10, self.y - 40))                

class ActionSpace(list):
    n = property(lambda self : len(self))

class BoatEnvironment:
    def __init__(self, dimensions, screen):
        self.boat1 = Boat((dimensions[0] // 2, dimensions[1] // 2), self, initial_speed=2, initial_angular_velocity=1, wrap=False)
        self.boat2 = Boat((dimensions[0] // 2, dimensions[1] // 2), self, initial_speed=2, initial_angular_velocity=-1)
        self.dimensions = dimensions
        self.action_space = ActionSpace([
            Boat.NOP,
            Boat.TURN_CW,
            Boat.TURN_CCW,
            Boat.INC_SPD,
            Boat.DEC_SPD
        ])
        self.observation_space = np.ndarray((4))
        self.screen = screen

    def step(self, action):
        return self.boat1.step(action)

    def reset(self):
        self.boat1 = Boat((self.dimensions[0] // 2, self.dimensions[1] // 2), self, initial_speed=2, initial_angular_velocity=1)
        return self.boat1.state

    def render(self):
        self.draw(self.screen)

    def draw(self, screen):
        # water
        pg.draw.rect(screen, (66, 84, 155), pg.Rect(0, 0, self.dimensions[0], self.dimensions[1]))
        
        # grid
        for i in range(self.dimensions[0] // 32):
            pg.draw.line(screen, (36, 54, 125), (i * 32, 0), (i * 32, self.dimensions[1]))
        for i in range(self.dimensions[1] // 32):
            pg.draw.line(screen, (36, 54, 125), (0, i * 32), (self.dimensions[0], i * 32))

        # boats
        self.boat1.draw(screen)
        self.boat2.draw(screen)

        text_surface, _ = font.render('1: spd, ang = {:.2f}, {:.2f}'.format(self.boat1.speed, self.boat1.angular_velocity), (0, 0, 0))
        screen.blit(text_surface, (32, 32))
        text_surface2, _ = font.render('2: spd, ang = {:.2f}, {:.2f}'.format(self.boat2.speed, self.boat2.angular_velocity), (0, 0, 0))
        screen.blit(text_surface2, (32, 96))