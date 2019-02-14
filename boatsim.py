import pygame as pg
import numpy as np
import matplotlib.pyplot as plt
import pygame.freetype as pgft
import math

myimage = pg.image.load("boat.png")
imagerect = myimage.get_rect()

# pygame setup
world_w, world_h = 640, 480
pg.init()
screen = pg.display.set_mode((world_w, world_h))
clock = pg.time.Clock()
font = pgft.SysFont('Comic Sans MS', 30)

i = 0
class Boat:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.w = 96
        self.h = 40
        self.direction = 0
        self.spd = 8
        self.angvel = 0

    def move(self):
        self.x += self.spd * math.cos(self.direction * ((math.pi * 2) / 360))
        self.y += self.spd * -math.sin(self.direction * ((math.pi * 2) / 360))
        self.direction += self.angvel
        
        if self.x > world_w: self.x = 0
        if self.x < 0: self.x = world_w
        if self.y > world_h: self.y = 0
        if self.y < 0: self.y = world_h
    
    def draw(self, screen):
        boat_image = pg.transform.rotate(myimage, self.direction)
        
        b_img_w, b_img_h = (boat_image.get_width(), boat_image.get_height())
        
        boat_pivot = (self.x - b_img_w / 2, self.y - b_img_h / 2)
        
        screen.blit(boat_image, imagerect.move(boat_pivot))
        
        pg.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), 4)
        
        line_len = 96
        # sensor 1
        pg.draw.line(screen, (0, 255, 0), (self.x, self.y), (self.x + line_len * math.cos((self.direction + 30) * ((math.pi * 2) / 360)), self.y + line_len * -math.sin((self.direction + 30) * ((math.pi * 2) / 360))))
        # sensor 2
        pg.draw.line(screen, (0, 255, 0), (self.x, self.y), (self.x + line_len * math.cos((self.direction - 30) * ((math.pi * 2) / 360)), self.y + line_len * -math.sin((self.direction - 30) * ((math.pi * 2) / 360))))
        

b_arr = []
for _ in range(10):
    b = Boat()
    b.angvel = np.random.uniform(-0.4, 0.4)
    b_arr.append(b)

framerate = 30

running = True
while running:
    # water
    pg.draw.rect(screen, (66, 84, 155), pg.Rect(0, 0, world_w, world_h))
    for i in range(world_w // 32):
        pg.draw.line(screen, (36, 54, 125), (i * 32, 0), (i * 32, world_h))
    for i in range(world_h // 32):
        pg.draw.line(screen, (36, 54, 125), (0, i * 32), (world_w, i * 32))

    for boat in b_arr:
        boat.move()
        boat.draw(screen)
    for event in pg.event.get():
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_SPACE:
                framerate = 1 if framerate != 1 else 30
        if event.type == pg.QUIT:
            pg.quit()
            running = False
    
    text_surface, rect = font.render('distance', (0, 0, 0))# = {:.2f}'.format(math.sqrt((b1.x - b2.x) ** 2 + (b1.y - b2.y) ** 2)), (0, 0, 0))
    screen.blit(text_surface, (40, 250))
    
    pg.display.flip()
    clock.tick(framerate)