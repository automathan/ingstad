import pygame as pg
import numpy as np
import pygame.freetype as pgft
import math

pg.init()
myimage = pg.image.load("boat.png")
imagerect = myimage.get_rect()
font = pgft.SysFont('Comic Sans MS', 28)
excl_mark, _ = font.render('!!!', (255, 0, 0))

def angular_offset(boat, target):
    vtarget = np.array([int(target[0]), int(target[1])])
    vboat = np.array([boat.x, boat.y])
    vdesired = vtarget - vboat
    thetaboat = math.atan2(math.sin(math.radians(boat.direction)), math.cos(math.radians(boat.direction)))
    thetatarget = math.atan2(vdesired[1], vdesired[0])
    thetadesired = math.degrees(thetatarget + thetaboat)
    if math.fabs(thetadesired) > math.fabs(thetadesired + 360):
        thetadesired += 360
    if math.fabs(thetadesired) > math.fabs(thetadesired - 360):
        thetadesired -= 360
    thetadesired = -thetadesired
    return int(thetadesired)
state_size = 6
class Boat:
    max_ship_spd = 1.5
    min_ship_spd = -0.5
    max_ang_vel = 1
    
    NOP, TURN_CW, TURN_CCW, INC_SPD, DEC_SPD, INC_SPD_CW, INC_SPD_CCW, DEC_SPD_CW, DEC_SPD_CCW  = range(9)
    def __init__(self, pos, env, initial_speed=0, initial_acceleration=0, initial_direction=0, initial_angular_velocity=0, wrap=False, goal_position=(200, 0)):
        self.x = pos[0]
        self.y = pos[1]
        self.length = 96 * 2
        self.speed = initial_speed
        self.acceleration = initial_acceleration
        self.direction = initial_direction
        self.angular_velocity = initial_angular_velocity
        self.env = env
        self.other_ship_dist = -1
        self.show_circles = False
        self.goal_position = goal_position
        #dist_to_goal = math.sqrt((self.x - self.goal_position[0]) ** 2 + (self.y - self.goal_position[1]) ** 2)
        off = angular_offset(self, self.goal_position)
        #self.prev_dist = dist_to_goal
        self.state = np.zeros(state_size)#np.ndarray(8)#[self.speed / self.max_ship_spd, self.angular_velocity / self.max_ang_vel, off / 180, dist_to_goal / self.env.dimensions[0]]
        # [speed, angular velocity, angular goal offset, angular other ship offset]
        # angular offset: how many radians away from heading straight towards a target [-pi, pi], negative is CW
        # self.wrap = wrap # wrap is practical for not going offscreen when using a human agent, but should be penalized in training

    def step(self, action):
        done = False
        if (action == Boat.INC_SPD or action == Boat.INC_SPD_CW or action == Boat.INC_SPD_CCW) and self.speed < Boat.max_ship_spd:
            self.speed += 0.05
        if (action == Boat.DEC_SPD or action == Boat.DEC_SPD_CW or action == Boat.DEC_SPD_CCW) and self.speed > Boat.min_ship_spd:
            self.speed -= 0.05
        if (action == Boat.TURN_CCW or action == Boat.INC_SPD_CCW or action == Boat.DEC_SPD_CCW) and self.angular_velocity < Boat.max_ang_vel:
            self.angular_velocity += 0.05
        if (action == Boat.TURN_CW or action == Boat.INC_SPD_CW or action == Boat.DEC_SPD_CW) and self.angular_velocity > -Boat.max_ang_vel:
            self.angular_velocity -= 0.05
        
        self.x += self.speed * math.cos(math.radians(self.direction))
        self.y += self.speed * -math.sin(math.radians(self.direction))
        if self.env.wrap:
            if self.x < 0: self.x = self.env.dimensions[0]
            if self.y < 0: self.y = self.env.dimensions[1]
            if self.x > self.env.dimensions[0]: self.x = 0
            if self.y > self.env.dimensions[1]: self.y = 0
        else:
            if self.x < 0 or self.y < 0 or self.x > self.env.dimensions[0] or self.y > self.env.dimensions[1]: 
                done = True
        
        self.speed += self.acceleration
        self.direction = (self.direction + self.angular_velocity) % 360
        self.other_ship_dist = math.sqrt((self.env.boat1.x - self.env.boat2.x) ** 2 + (self.env.boat1.y - self.env.boat2.y) ** 2)
        dist_to_goal = math.sqrt((self.x - self.goal_position[0]) ** 2 + (self.y - self.goal_position[1]) ** 2)
        reward = 0 #int(2 * (self.prev_dist - dist_to_goal)) - 1
        #self.prev_dist = dist_to_goal
        
        goal_off = angular_offset(self, self.goal_position)
        ship_off = angular_offset(self, (self.env.boat2.x, self.env.boat2.y))
        
        self.state = [self.speed / Boat.max_ship_spd, self.angular_velocity / Boat.max_ang_vel, goal_off / 180, dist_to_goal / self.env.dimensions[0], self.other_ship_dist / self.env.dimensions[0], ship_off / 180]        
        
        if self.other_ship_dist < self.length / 3:
            reward = -1
            done = True

        if dist_to_goal < 50:
            reward = 1 if self.speed > 0 else 0.5
            done = True
        
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
        pg.draw.circle(screen, (0, 255, 0), (int(self.goal_position[0]), int(self.goal_position[1])), 50, 1)
        pg.draw.line(screen, (255, 0, 0), (self.env.boat1.x, self.env.boat1.y), (self.env.boat2.x, self.env.boat2.y))
        pg.draw.line(screen, (255, 0, 0), (self.x, self.y), (self.x + self.length * math.cos(math.radians(self.direction)), self.y + self.length * -math.sin(math.radians(self.direction))))
        pg.draw.line(screen, (255, 0, 0), (self.x, self.y), self.goal_position)
        if self.other_ship_dist < self.length / 3:
            screen.blit(excl_mark, (self.x - 10, self.y - 40))                

class ActionSpace(list):
    n = property(lambda self : len(self))

class BoatEnvironment:
    def __init__(self, dimensions, screen, wrap=False):
        self.dimensions = dimensions
        self.action_space = ActionSpace([
            Boat.NOP,
            Boat.TURN_CW,
            Boat.TURN_CCW,
            Boat.INC_SPD,
            Boat.DEC_SPD,
            Boat.INC_SPD_CW,
            Boat.INC_SPD_CCW,
            Boat.DEC_SPD_CW,
            Boat.DEC_SPD_CCW
        ])
        self.observation_space = np.ndarray(state_size)
        self.screen = screen
        self.boat1 = Boat((np.random.uniform(high=self.dimensions[0]), np.random.uniform(high=self.dimensions[1])), self, initial_speed=0, initial_angular_velocity=0, wrap=False)
        self.boat2 = Boat((dimensions[0] // 2, dimensions[1] // 2), self, initial_speed=0, initial_angular_velocity=0)
        self.wrap = wrap
        self.reset()

    def step(self, action):
        return self.boat1.step(action)

    def reset(self):
        self.boat1 = Boat((self.dimensions[0] // 2, self.dimensions[1] // 2), self, wrap=False, initial_direction=np.random.uniform(0, 360), initial_speed=np.random.uniform(Boat.min_ship_spd, Boat.max_ship_spd))
        self.boat2 = Boat((self.dimensions[0] // 2, self.dimensions[1] // 2 + 100), self, wrap=False, initial_direction=np.random.uniform(0, 360))
        return self.boat1.state

    def render(self):
        self.draw(self.screen)
        pg.display.flip()

    def draw(self, screen):
        # water
        pg.draw.rect(screen, (66, 84, 155), pg.Rect(0, 0, self.dimensions[0], self.dimensions[1]))
        
        # grid
        for i in range(self.dimensions[0] // 32): pg.draw.line(screen, (36, 54, 125), (i * 32, 0), (i * 32, self.dimensions[1]))
        for i in range(self.dimensions[1] // 32): pg.draw.line(screen, (36, 54, 125), (0, i * 32), (self.dimensions[0], i * 32))

        self.boat1.draw(screen)
        self.boat2.draw(screen)

        text_surface, _ = font.render('1: spd, ang = {:.2f}, {:.2f}'.format(self.boat1.speed, self.boat1.angular_velocity), (0, 0, 0))
        screen.blit(text_surface, (32, 32))