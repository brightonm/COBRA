import pygame
import random
from config import Point, Direction, CONFIG
import numpy as np


# rgb colors
WHITE, BLACK, GREEN, RED = CONFIG["colors"].values()


class RLCobraEnvironment:

    def __init__(self):
        self.w = CONFIG["game"]["width"]
        self.h = CONFIG["game"]["height"]
        self.speed = CONFIG["game"]["speed"]
        self.block_size = CONFIG["game"]["block_size"]
        
        # init display
        pygame.init()
        self.font = pygame.font.Font("assets/Now-Regular.otf", 25)
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Cobra')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # choose a random direction
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head]

        self.score = 0
        self.food = None
        self.regenerate_apple()
        self.frame_iteration = 0


    def regenerate_apple(self):
        x = random.randint(0, (self.w-self.block_size )//self.block_size )*self.block_size
        y = random.randint(0, (self.h-self.block_size )//self.block_size )*self.block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self.regenerate_apple()


    def play_step(self, action, n_games, record):
        self.frame_iteration += 1
        
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self.move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.regenerate_apple()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self.update_game(n_games, record)
        self.clock.tick(self.speed)
        # 6. return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - self.block_size or pt.x < 0 or pt.y > self.h - self.block_size or pt.y < 0:
            return True
        
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False


    def update_game(self, n_games, record):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
            pygame.draw.rect(self.display, BLACK, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size))
        
        
        text_score = self.font.render(f"Score: {self.score}", True, WHITE)
        text_title = self.font.render("COBRA", True, WHITE)
        text_n_games = self.font.render(f"Number of Games: {n_games}", True, WHITE)
        text_highest_score = self.font.render(f"Highest: {record}", True, WHITE)
        self.display.blit(text_title, [self.w / 2 - self.block_size * 0.5, 0])
        self.display.blit(text_score, [0, 0])
        self.display.blit(text_highest_score, [0, self.block_size * 1.5])
        self.display.blit(text_n_games, [0, self.h - self.block_size * 2])
        pygame.display.flip()


    def move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size

        self.head = Point(x, y)