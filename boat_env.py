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
        self.length = 96 * 2
        self.speed = initial_speed
        self.acceleration = initial_acceleration
        self.direction = initial_direction
        self.angular_velocity = initial_angular_velocity
        self.env = env
        self.other_ship_dist = -1
        self.max_ship_spd = 8
        self.min_ship_spd = -3
        self.max_ang_vel = 2
        self.show_circles = False
        self.goal_position = (300, 300)
        
        self.state = [self.x / self.env.dimensions[0], self.y / self.env.dimensions[1], math.radians(self.direction), self.speed / self.max_ship_spd]
        # [speed, angular velocity, angular goal offset, angular other ship offset]
        # angular offset: how many radians away from heading straight towards a target [-pi, pi], negative is CW

        self.wrap = wrap # wrap is practical for not going offscreen when using a human agent, but should be penalized in training

    def step(self, action):
        done = False

        if action == Boat.INC_SPD and self.speed < self.max_ship_spd:
            self.speed += 0.1
        if action == Boat.DEC_SPD and self.speed > self.min_ship_spd:
            self.speed -= 0.1
        if action == Boat.TURN_CCW and self.angular_velocity < self.max_ang_vel:
            self.angular_velocity += 0.1
        if action == Boat.TURN_CW and self.angular_velocity > -self.max_ang_vel:
            self.angular_velocity -= 0.1

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
        self.direction = (self.direction + self.angular_velocity) % 360
        
        self.other_ship_dist = math.sqrt((self.env.boat1.x - self.env.boat2.x) ** 2 + (self.env.boat1.y - self.env.boat2.y) ** 2)
        dist_to_goal = math.sqrt((self.x - self.goal_position[0]) ** 2 + (self.y - self.goal_position[1]) ** 2)
        

        self.state = [self.x / self.env.dimensions[0], self.y / self.env.dimensions[1], math.radians(self.direction), self.speed / self.max_ship_spd]
        reward = -1
        
        if dist_to_goal < 50:
            reward = 100
        
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
        
        #self.boat1 = Boat((self.dimensions[0] // 2, self.dimensions[1] // 2), self, initial_speed=0, initial_angular_velocity=0, wrap=False)
        self.boat1 = Boat((np.random.uniform(high=self.dimensions[0]), np.random.uniform(high=self.dimensions[1])), self, initial_speed=0, initial_angular_velocity=0, wrap=False)
        self.boat2 = Boat((dimensions[0] // 2, dimensions[1] // 2), self, initial_speed=0, initial_angular_velocity=0)

    def step(self, action):
        return self.boat1.step(action)

    def reset(self):
        #self.boat1 = Boat((self.dimensions[0] // 2, self.dimensions[1] // 2), self, initial_speed=0, initial_angular_velocity=0, wrap=False)
        self.boat1 = Boat((np.random.uniform(high=self.dimensions[0]), np.random.uniform(high=self.dimensions[1])), self, initial_speed=0, initial_angular_velocity=0, wrap=False)
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

        """
        #vtarget=np.array([500,400])
        vtarget=np.array([int(self.boat1.x),int(self.boat1.y)])
        vboat=np.array([self.boat2.x,self.boat2.y])
        vdesired = vtarget-vboat
        #print(vdesired)
        thetaboat=math.atan2(math.sin(self.boat2.direction*math.pi/180),math.cos(self.boat2.direction*math.pi/180))*180/math.pi
        thetatarget=math.atan2(vdesired[1],vdesired[0])*180/math.pi
        
        thetadesired=thetatarget+thetaboat
        if math.fabs(thetadesired)>math.fabs(thetadesired+360):
            thetadesired=thetadesired+360
        if math.fabs(thetadesired)>math.fabs(thetadesired-360):
            thetadesired=thetadesired-360
        thetadesired=-thetadesired

        pg.draw.circle(screen, (255, 250, 0), vtarget, 10, 1)
        print(int(thetaboat),int(thetatarget))
        """

        print(math.radians(self.boat1.direction), math.radians(self.boat2.direction))
        # boats
        self.boat1.draw(screen)
        self.boat2.draw(screen)

        text_surface, _ = font.render('1: spd, ang = {:.2f}, {:.2f}'.format(self.boat1.speed, self.boat1.angular_velocity), (0, 0, 0))
        screen.blit(text_surface, (32, 32))
        text_surface2, _ = font.render('2: spd, ang = {:.2f}, {:.2f}'.format(self.boat2.speed, self.boat2.angular_velocity), (0, 0, 0))
        screen.blit(text_surface2, (32, 96))