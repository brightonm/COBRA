import pygame
import random
from config import Point, Direction, CONFIG

WHITE, BLACK, GREEN, RED = CONFIG["colors"].values()


class CobraGame:
    
    def __init__(self):
        self.w = CONFIG["game"]["width"]
        self.h = CONFIG["game"]["height"]
        self.speed = CONFIG["game"]["speed"] / 2
        self.block_size = CONFIG["game"]["block_size"]
        
        # init display
        pygame.init()
        self.font = pygame.font.Font("assets/Now-Regular.otf", 25)
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('COBRA')
        self.clock = pygame.time.Clock()
        
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head]
        
        self.score = 0
        self.food = None
        self.regenerate_apple()
        
    def regenerate_apple(self):
        x = random.randint(0, (self.w-self.block_size )//self.block_size )*self.block_size 
        y = random.randint(0, (self.h-self.block_size )//self.block_size )*self.block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self.regenerate_apple()
        
    def play_step(self, n_games, record):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        
        # 2. move
        self.move(self.direction) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        game_over = False
        if self.is_collision():
            game_over = True
            return game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            self.regenerate_apple()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self.update_game(n_games, record)
        self.clock.tick(self.speed)
        # 6. return game over and score
        return game_over, self.score
    
    def is_collision(self):
        # hits boundary
        if self.head.x > self.w - self.block_size or self.head.x < 0 or self.head.y > self.h - self.block_size or self.head.y < 0:
            return True
        # hits itself
        if self.head in self.snake[1:]:
            return True
        
        return False
        
    def update_game(self, n_games, record):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
            pygame.draw.rect(self.display, BLACK, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size))
        
        text_score = self.font.render(f"Score: {self.score}", True, WHITE)
        text_title = self.font.render("Cobra", True, WHITE)
        text_n_games = self.font.render(f"Number of Games: {n_games}", True, WHITE)
        text_highest_score = self.font.render(f"Highest: {record}", True, WHITE)
        self.display.blit(text_title, [self.w / 2 - self.block_size * 0.5, 0])
        self.display.blit(text_score, [0, 0])
        self.display.blit(text_highest_score, [0, self.block_size * 1.5])
        self.display.blit(text_n_games, [0, self.h - self.block_size * 2])
        pygame.display.flip()
        
    def move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += self.block_size
        elif direction == Direction.LEFT:
            x -= self.block_size
        elif direction == Direction.DOWN:
            y += self.block_size
        elif direction == Direction.UP:
            y -= self.block_size
            
        self.head = Point(x, y)
            

if __name__ == '__main__':
    
    record = 0
    n_games = 0
    # games loop
    while True:
        game = CobraGame()
        # game loop
        while True:
            game_over, score = game.play_step(n_games, record)
            
            if game_over == True:
                n_games += 1
                if score > record :
                    record = score
                break
            
        pygame.quit()