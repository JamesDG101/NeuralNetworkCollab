import pygame as pg
import math
pg.init()



class Window:
    SIZE = 28

    def __init__(self):
        self.window = pg.display.set_mode((1200,644))
        pg.display.set_caption("GUI")
        self.window.fill("white")

    def draw_grid(self,square_size):
        start = (0,0)
        

        x_pos = start[0]
        y_pos = start[1]
        for square_x in range(self.SIZE):
            for square_y in range(self.SIZE):
                position = (square_x*square_size,square_y*square_size,square_size,square_size)
                pg.draw.rect(self.window,"black",position,1)
    
    def draw(self):
        mouse_pos = pg.mouse.get_pos()
        if mouse_pos[0] <= 644 and mouse_pos[1] <= 644:
            mouse_square = (math.ceil(mouse_pos[0]/23),math.ceil(mouse_pos[1]/23))
            print(mouse_square)
        
        
           





w = Window()
w.draw_grid(23)
running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        
    if pg.mouse.get_pressed()[0]:
        w.draw()
        

   
    pg.display.flip()




pg.quit()