import pygame
import neat
import time
import pickle
import os
import random
pygame.font.init()


WINDOW_WIDTH = 500
WINDOW_HEIGHT = 800

DINO_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","tyrannosaurus" + str(x) + ".png"))) for x in range(1,4)]
VOLCANO_IMGS = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "volcano.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "sky.jpg")))

STAT_FONT = pygame.font.SysFont("comicsans", 50)

class Dino:
    MAX_ROTATION = 25
    IMGS = DINO_IMGS
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel =-10.5
        self.tick_count = 0
        self.height - self.y

    def move(self):
        self.tick_count += 1

        d = self.vel*self.tick_count + 1.5 *self.tick_count**2

        if d >= 16:
            d = 16
        
        if d<0:
            d-=2

        self.y = self.y + d

        # if d < 0 or self.y < self.height + 50:
        #     if self.tilt < self.MAX_ROTATION:
        #         self.tilt = self.MAX_ROTATION
        # else:
        #     if self.tilt >-90:
        #         self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1

        if self.img_count <= self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0
        
        if self.tilt<=-80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2
        
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center = self.img.get_rect(topleft = (self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)
    
    def get_mask(self):
        return pygame.mask.from_surface(self.img)
    

class Volcano:
    GAP = 200
    VEL = 5

    def __init__(self,x):
        self.x = x
        self.height = 0
        self.gap = 100

        self.top = 0
        self.bottom = 0
        self.VOLCANO_TOP = pygame.transform.flip(VOLCANO_IMGS, False,True)
        self.Volcano_Bottom = VOLCANO_IMGS

        self.passed = False
        self.set_height()
    def set_height(self):
        self.height = random.randrange(50,450)
        self.top = self.height - self.VOLCANO_TOP.get_height()
        self.bottom = self.height + self.GAP
    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.VOLCANO_TOP, (self.x, self.top))
        win.blit(self.Volcano_Bottom,(self.x,self.bottom))
    
    def collide(self, dino):
        dino_mask = dino.get_mask()
        top_mask = pygame.mask.from_surface(self.VOLCANO_TOP)
        bottom_mask = pygame.mask.from_surface(self.Volcano_Bottom)

        top_offset = (self.x - dino.x, self.top - round(dino.y))
        bottom_offset = (self.x-dino.x, self.bottom - round(dino.y))

        b_point = dino_mask.overlap(bottom_mask, bottom_offset)
        t_point = dino_mask.overlap(top_mask,top_offset)

        if t_point or b_point:
            return True
        return False
class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

# Moving volcano image circle  of images
    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH <0:
            self.x1 = self.x2 + self.WIDTH
        
        if self.x2 + self.WIDTH <0:
            self.x2 = self.x1 + self.WIDTH
    
    def draw (self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))




    
def draw_window(win,dinos, volcanos, base, score):
    win.blit(BG_IMG, (0,0))
    for volcano in volcanos:
        volcano.draw(win)
    

    text = STAT_FONT.render("Score: " + str(score), 1,(255,255,255))
    win.blit(text, (WINDOW_WIDTH-10 - text.get_width(), 10) )

    base.draw(win)

    for dino in dinos:
        dino.draw(win)

    pygame.display.update()

def main(genomes, config):
    nets = []
    ge = []
    dinos = []

    for _,g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        dinos.append(Dino(230, 350))
        g.fitness = 0
        ge.append(g)


    base = Base(730)
    volcanos = [Volcano(600)]
    win = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    score = 0

    run = True

    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
    
    #if passed firt volcano
        volcano_ind = 0
        if len(dinos) > 0:
            if len(volcanos) >1 and dinos[0].x > volcanos[0].x + volcanos[0].VOLCANO_TOP.get_width():
                volcano_ind = 1
        else:
            run = False
            break
        
        for x, dino in enumerate(dinos):
            dino.move()
            ge[x].fitness += 0.1

            output = nets[x].activate((dino.y, abs(dino.y - volcanos[volcano_ind].height), abs(dino.y - volcanos[volcano_ind].bottom)))

            if output[0] > 0.5:
                dino.jump()


        add_volcano = False        
        rem = []
        #removing dinos once collision takes place
        for volcano in volcanos:
            for x, dino in enumerate(dinos):
                if volcano.collide(dino):
                    ge[x].fitness -= 1
                    dinos.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not volcano.passed and volcano.x < dino.x:
                    volcano.passed = True
                    add_volcano = True
            if volcano.x + volcano.VOLCANO_TOP.get_width() < 0:
                rem.append(volcano)

            volcano.move()

        if add_volcano:
            score += 1
            for g in ge:
                g.fitness += 5
            volcanos.append(Volcano(600))

        for r in rem:
            volcanos.remove(r)
        
        #checks to see if dinos hit the ground
        for x, dino in enumerate(dinos):
            if dino.y + dino.img.get_height() >= 730 or dino.y < 0:
                dinos.pop(x)
                nets.pop(x)
                ge.pop(x)



        base.move()
        draw_window(win,dinos, volcanos, base, score)

  

def run(config_path):
     config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
     # setting the population
     p = neat.Population(config)

    #providing the output stats reporters shows detailed information
     p.add_reporter(neat.StdOutReporter(True))
     stats = neat.StatisticsReporter()
     p.add_reporter(stats)

    #runs main loop 50 times
     winner = p.run(main,50)




if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run (config_path)




        


