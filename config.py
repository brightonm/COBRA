import pygame
from enum import Enum
from collections import namedtuple

CONFIG = {
    'colors': 
        {
        'white':  (255, 255, 255),
        'black': (0, 0, 0),
        'green': (0, 128, 0),
        'red': (100, 0, 0)
        },
    'game':
        {
        'width': 640,      
        'height': 480,
        'block_size': 20,
        'speed': 40
        },
    
}


Point = namedtuple('Point', 'x, y')

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
pygame.init()
font = pygame.font.Font("assets/Now-Regular.otf", 25)
