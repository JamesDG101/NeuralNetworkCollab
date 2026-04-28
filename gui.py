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
        
            mouse_square = [mouse_square[0]-1,mouse_square[1]-1]
            position = [mouse_square[0]*23,mouse_square[1]*23,23,23]
            pg.draw.rect(self.window,"black",position)

    def draw_text(self,text,pos,size):
        indent = size//6 + 5
        pos = [pos[0]+indent,pos[1]+indent]
        font = pg.font.SysFont(None,size)
        text = font.render(text,True,"black")
        self.window.blit(text,(pos))
    

    def draw_num(self,num):
        position = (800,100,220,300)
        pg.draw.rect(self.window,"black",position,10)
        self.draw_text(str(num),position,300)


    def handle_clear():
        pass




class Button:
    def __init__(self,pos,size,text=None,text_size=None):
        self.size = size
        self.pos = pos
        self.text = text
        self.text_size = text_size

        self.rect = pg.Rect(pos[0],pos[1],size[0],size[1])

    def is_clicked(self,pos):
        if self.rect.collidepoint(pos):
            w.window.fill("white")
            w.draw_grid(SQUARE_SIZE)
            self.draw_btn()
            w.draw_num(9)

    def draw_btn(self):

        w.draw_text(self.text,self.pos,self.text_size)

        pg.draw.rect(w.window,"black",(self.pos[0],self.pos[1],self.size[0],self.size[1]),5)



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