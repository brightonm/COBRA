import os
from pathlib import Path
import torch
import random
import argparse
import numpy as np
from collections import deque
from rl_environment import RLCobraEnvironment, Direction, Point
from model import AgentTrainer, build_neural_net, DEVICE

# Fix seed for reproducing
torch.manual_seed(7)
random.seed(7)
np.random.seed(7)


MAX_MEMORY = 100_000
BATCH_SIZE = 1024

def save(model, game, score):
    models_folder_path = Path("models/")
    if not os.path.exists(models_folder_path):
        os.makedirs(models_folder_path)

    filepath = models_folder_path / Path(f"model_game_{game}_score_{score}.pth")
    torch.save(model.state_dict(), filepath)

class RLAgent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = build_neural_net()
        self.learning_rate = 0.001
        self.trainer = AgentTrainer(self.model, lr=self.learning_rate, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, is_pretrained_model=False):
        
        final_move = [0,0,0]
        if not is_pretrained_model:
        # random moves: tradeoff exploration / exploitation
            self.epsilon = 80 - self.n_games
            if random.randint(0, 200) < self.epsilon:
                move = random.randint(0, 2)
                final_move[move] = 1
            else:
                state0 = torch.tensor(state, dtype=torch.float, device=DEVICE)
                prediction = self.model(state0).cpu()
                move = torch.argmax(prediction).item()
                final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float, device=DEVICE)
            prediction = self.model(state0).cpu()
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train(args):
    record = 0
    agent = RLAgent()
    environment = RLCobraEnvironment()
    
    pretrained_model_path = args["model_path"]
    if pretrained_model_path is not None:
        agent.model.load_state_dict(torch.load(pretrained_model_path))
        
    is_pretrained_model = pretrained_model_path is not None
    
    while True:
        # get old state
        state_old = agent.get_state(environment)

        # get move
        final_move = agent.get_action(state_old, is_pretrained_model)

        # perform move and get new state
        reward, dead, score = environment.play_step(final_move, agent.n_games, record)
        state_new = agent.get_state(environment)

        if pretrained_model_path is None:
            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, dead)

            # remember
            agent.remember(state_old, final_move, reward, state_new, dead)

        if dead:
            # train long memory, plot result
            environment.reset()
            agent.n_games += 1
            if pretrained_model_path is None:
                agent.train_long_memory()

            if score > record:
                record = score
                if pretrained_model_path is None:
                    save(agent.model, agent.n_games, record)

            print('Game', agent.n_games, 'Score', score, 'Record:', record)


def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COBRA : AI agent trained on Snake Game environment')
    parser.add_argument('--model_path', help='Path to the model we want to reuse', type=file_path, required=False)
    args = vars(parser.parse_args())
    train(args)