import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import time



class Board:

    def __init__(self):

        self.b = None
        self.x = None
        self.y = None
        self.complete = False

        self.nx = 28
        self.ny = 28
        
        self.fig = plt.figure()
        self.img = np.zeros((self.nx, self.ny))
        self.im = plt.imshow(self.img, cmap='gist_gray_r', vmin=0, vmax=1)
        
        plt.connect('motion_notify_event', self.mouse_move)

        self.anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init, frames=self.nx*self.ny, interval=50)    


    def mouse_move(self, event):

        self.x, self.y, self.b = event.xdata, event.ydata, event.button



    def setPixel(self, img, x, y, val):
        if x>=0 and y>=0 and x < img.shape[0] and y < img.shape[1]:
            img[x, y] += val 
            if img[x, y] > 1:
                img[x, y] = 1


    def drawPoint(self, img, x, y):
        
        self.setPixel(img, x, y, 1)
        self.setPixel(img, x+1, y, 0.25)
        self.setPixel(img, x, y+1, 0.25)
        self.setPixel(img, x-1, y+1, 0.25)
        self.setPixel(img, x+1, y+1, 0.25)
            
        self.setPixel(img, x+1, y, 0.25)
        self.setPixel(img, x, y+1, 0.25)
        self.setPixel(img, x-1, y, 0.25)
        self.setPixel(img, x, y-1, 0.25)






    def init(self):
        self.im.set_data(np.zeros((self.nx, self.ny)))
    
    def animate(self, i):
        
        
        if self.b == plt.MouseButton.LEFT:
            self.drawPoint(self.img, int(self.y), int(self.x))
            
        if self.b == plt.MouseButton.RIGHT:
            plt.close()
            self.complete = True
            
    
        self.im.set_data(self.img + np.random.randn(self.nx,self.ny)*0.05)
        return self.im


    def run(self):
        
        while not self.complete:
            plt.pause(0.1)
            
        plt.close()









