import pygame as pg
import numpy as np
import pygame.freetype as pgft
import math

pg.init()
myimage = pg.image.load('res/boat.png')
imagerect = myimage.get_rect()
font = pgft.SysFont('Comic Sans MS', 28)
excl_mark, _ = font.render('!!!', (255, 0, 0))

def angular_offset(boat, target):
    vtarget = np.array(target)
    vboat = boat.position

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

state_size = 9

class Boat:
    max_ship_spd = 1
    min_ship_spd = -0.5
    max_ang_vel = 0.3
    ship_length = 192

    NOP, TURN_CW, TURN_CCW, INC_SPD, DEC_SPD, INC_SPD_CW, INC_SPD_CCW, DEC_SPD_CW, DEC_SPD_CCW  = range(9)
    def __init__(self, pos, env, other_ship, initial_speed=0, initial_direction=0, initial_angular_velocity=0, goal_position=(200, 0)):
        self.env = env

        # Boat constants
        self.length = Boat.ship_length

        # Internal boat state variables
        self.x, self.y = pos
        self.speed = initial_speed
        self.direction = initial_direction
        self.angular_velocity = initial_angular_velocity
        
        # Episode dependent variables
        self.goal_position = goal_position
        self.other_ship = other_ship
        self.show_circles = False

    position = property(fget=lambda self : (self.x, self.y))

    def update_state(self):
        other_ship_dist = math.hypot(self.x - self.other_ship.x, self.y - self.other_ship.y)
        dist_to_goal = math.hypot(self.x - self.goal_position[0], self.y - self.goal_position[1])
        
        self.prev_dist = dist_to_goal
        goal_offset = angular_offset(self, self.goal_position)
        collision_offset = angular_offset(self, self.other_ship.position)
        
        other_goal_offset = angular_offset(self.other_ship, self.other_ship.goal_position)
        other_collision_offset = angular_offset(self.other_ship, self.position)

        self.state = [
            self.speed / Boat.max_ship_spd, 
            self.angular_velocity / Boat.max_ang_vel, 
            goal_offset / 180, 
            dist_to_goal / self.env.dimensions[0], 
            other_ship_dist / self.env.dimensions[0], 
            collision_offset / 180, 
            other_goal_offset / 180, 
            other_collision_offset / 180, 
            self.other_ship.speed
        ]       
        
    def step(self, action):
        done = False
        if (action == Boat.INC_SPD or action == Boat.INC_SPD_CW or action == Boat.INC_SPD_CCW) and self.speed < Boat.max_ship_spd: self.speed += 0.05
        if (action == Boat.DEC_SPD or action == Boat.DEC_SPD_CW or action == Boat.DEC_SPD_CCW) and self.speed > Boat.min_ship_spd: self.speed -= 0.05
        if (action == Boat.TURN_CCW or action == Boat.INC_SPD_CCW or action == Boat.DEC_SPD_CCW) and self.angular_velocity < Boat.max_ang_vel: self.angular_velocity += 0.05
        if (action == Boat.TURN_CW or action == Boat.INC_SPD_CW or action == Boat.DEC_SPD_CW) and self.angular_velocity > -Boat.max_ang_vel:   self.angular_velocity -= 0.05
        
        self.x += self.speed * math.cos(math.radians(self.direction))
        self.y += self.speed * -math.sin(math.radians(self.direction))
        self.direction = (self.direction + self.angular_velocity) % 360
        
        other_ship_dist = math.hypot(self.x - self.other_ship.x, self.y - self.other_ship.y)
        dist_to_goal = math.hypot(self.x - self.goal_position[0], self.y - self.goal_position[1])
        
        reward = float(self.prev_dist - dist_to_goal) if self.env.intermediate_rewards else -1

        self.update_state()

        # Collision
        if other_ship_dist < self.length / 3:
            reward = -1000
            done = True

        # Goal
        if dist_to_goal < 50:
            reward = 1000 if self.speed > 0 else 100
            done = True
        
        return np.array(self.state), float(reward), done, {}

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
        pg.draw.line(screen, (255, 0, 0), (self.x, self.y), (self.other_ship.x, self.other_ship.y))
        pg.draw.line(screen, (255, 0, 0), (self.x, self.y), (self.x + 32 * self.length * math.cos(math.radians(self.direction)), self.y + 32 * self.length * -math.sin(math.radians(self.direction))))
        pg.draw.line(screen, (255, 0, 0), (self.x, self.y), self.goal_position)
        
        other_ship_dist = math.hypot(self.x - self.other_ship.x, self.y - self.other_ship.y)
        if other_ship_dist < self.length / 3:
            screen.blit(excl_mark, (self.x - 10, self.y - 40))    

    def reset(self, position, speed, direction, goal_position):
        self.x, self.y = position
        self.speed = speed
        self.direction = direction
        self.goal_position = goal_position

class ActionSpace(list):
    n = property(lambda self : len(self))

class BoatEnvironment:
    def __init__(self, dimensions, screen, intermediate_rewards=False, multi_agent=True):
        # Environmental parameters
        self.dimensions = dimensions

        # RL parameters
        self.observation_space = np.ndarray(state_size)
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

        self.intermediate_rewards = intermediate_rewards
        
        # Agents
        self.boat1 = Boat((dimensions[0] // 2, dimensions[1]), self, None)
        self.boat2 = Boat((dimensions[0], dimensions[1] // 2), self, self.boat1)
        self.boat1.other_ship = self.boat2

        # Misc
        self.multi_agent = multi_agent
        self.screen = screen
        self.reset()

    def step(self, action):
        self.boat2.step(Boat.NOP)
        return self.boat1.step(action)

    def reset(self, randomize=True):
        if randomize:
            self.boat1.reset(
                (np.random.uniform(self.dimensions[0] * 0.1, self.dimensions[0] * 0.9), np.random.uniform(self.dimensions[1] * 0.1, self.dimensions[1] * 0.9)), 
                np.random.uniform(Boat.min_ship_spd, Boat.max_ship_spd), 
                np.random.uniform(0, 360), 
                (np.random.uniform(self.dimensions[0] * 0.1, self.dimensions[0] * 0.9), np.random.uniform(self.dimensions[1] * 0.1, self.dimensions[1] * 0.9))
            )
            self.boat2.reset(
                (np.random.uniform(self.dimensions[0] * 0.1, self.dimensions[0] * 0.9), np.random.uniform(self.dimensions[1] * 0.1, self.dimensions[1] * 0.9)), 
                np.random.uniform(Boat.min_ship_spd, Boat.max_ship_spd), 
                np.random.uniform(0, 360), 
                (np.random.uniform(self.dimensions[0] * 0.1, self.dimensions[0] * 0.9), np.random.uniform(self.dimensions[1] * 0.1, self.dimensions[1] * 0.9))
            )
        else:
            self.boat1.reset((self.dimensions[0] // 2, self.dimensions[1]), np.random.uniform(0, Boat.max_ship_spd), 90, (np.random.uniform(self.dimensions[0] * 0.2, self.dimensions[0] * 0.8), 0))
            self.boat2.reset((self.dimensions[0], self.dimensions[1] // 2), np.random.uniform(0, Boat.max_ship_spd) if self.multi_agent else 0, 180, (0, np.random.uniform(self.dimensions[1] * 0.2, self.dimensions[1] * 0.8)))
        
        self.boat1.update_state()
        self.boat2.update_state()

        return np.array(self.boat1.state)

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