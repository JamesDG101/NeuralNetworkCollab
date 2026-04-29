import pygame as pg
import numpy as np

from network_2 import Network

pg.init()

class Window:
    # Initialise the window class that manages input and majority of the interface
    SIZE = 28

    def __init__(self):
        self.window = pg.display.set_mode((1200,644))
        pg.display.set_caption("GUI")
        self.window.fill("white")

        self.inputs = np.zeros(((self.SIZE ** 2), 1))
        self.curr_guess = 0

        self.net = Network(
            sizes=[self.SIZE ** 2, 30, 10],
            training_data_dir='data/train-images.gz',
            training_labels_dir='data/train-labels.gz',
            test_data_dir='data/test-images.gz',
            test_labels_dir='data/test-labels.gz',
            load_from='weights/saved3-30.npz',
        )

    def draw_grid(self,square_size): # Draws grid based on resolution of images and size of each pixel
        for square_x in range(self.SIZE):
            for square_y in range(self.SIZE):
                position = (square_x*square_size,square_y*square_size,square_size,square_size)
                pg.draw.rect(self.window,"black",position,1)

    def draw(self): 
        def draw_sq(grid_x, grid_y, color="black"):# Fills in correct cell when mouse is dragged over
            position = (grid_x * SQUARE_SIZE, grid_y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pg.draw.rect(self.window, color, position)
        
        mouse_pos = pg.mouse.get_pos()
        if 0 <= mouse_pos[0] < self.SIZE * SQUARE_SIZE and 0 <= mouse_pos[1] < self.SIZE * SQUARE_SIZE:
            grid_x = mouse_pos[0] //SQUARE_SIZE
            grid_y = mouse_pos[1] //SQUARE_SIZE

            draw_sq(grid_x, grid_y)

            idx = grid_y * self.SIZE + grid_x
            self.inputs[idx, 0] = 1

            output = self.net.feedforward(self.inputs)
            self.curr_guess = int(np.argmax(output))
            w.draw_num(self.curr_guess)

    def draw_text(self,text,pos,size): # Basic function to draw text to window
        indent = size//6 + 5
        pos = [pos[0]+indent,pos[1]+indent]
        font = pg.font.SysFont(None,size)
        text = font.render(text,True,"black")
        self.window.blit(text,(pos))
    

    def draw_num(self,num): # Function that draws number with highest probability
        position = (800,100,220,300)
        pg.draw.rect(self.window, "white", position)
        pg.draw.rect(self.window, "black", position, 10)
        self.draw_text(str(num), position, 300)

    def handle_clear(self): # Reset data inputs
        self.inputs = np.zeros(((self.SIZE ** 2), 1))
        self.curr_guess = 0

class Button:
    def __init__(self,pos,size,text=None,text_size=None):
        self.size = size
        self.pos = pos
        self.text = text
        self.text_size = text_size

        self.rect = pg.Rect(pos[0],pos[1],size[0],size[1])

    def is_clicked(self,pos): # Handles graphics when clear button activated
        if self.rect.collidepoint(pos):
            w.handle_clear()
            w.window.fill("white")
            w.draw_grid(SQUARE_SIZE)
            self.draw_btn()
            w.draw_num(w.curr_guess)

    def draw_btn(self):
        pg.draw.rect(w.window, "black", (self.pos[0], self.pos[1], self.size[0], self.size[1]), 5)
        w.draw_text(self.text, self.pos, self.text_size)

SQUARE_SIZE = 23
w = Window()
w.draw_grid(SQUARE_SIZE)

clear_btn = Button((700,500),(120,70),"Clear",36)
clear_btn.draw_btn()
running = True
w.draw_num(9)
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if event.type == pg.MOUSEBUTTONDOWN:
            clear_btn.is_clicked(event.pos)
        
    if pg.mouse.get_pressed()[0]:
        w.draw()
  
    pg.display.flip()

pg.quit()
