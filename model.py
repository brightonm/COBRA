import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
def build_neural_net():
    return nn.Sequential(
            nn.Linear(11, 256), 
            nn.ReLU(),
            nn.Linear(256, 3),
    ).to(DEVICE)


class AgentTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, dead):
        state = torch.tensor(state, dtype=torch.float, device=DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float, device=DEVICE)
        action = torch.tensor(action, dtype=torch.long, device=DEVICE)
        reward = torch.tensor(reward, dtype=torch.float, device=DEVICE)
        # (batch_size, dim)

        if len(state.shape) == 1:
            # (batch_size, dim)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            dead = (dead, )

        # 1: predicted Q values with current state and action
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(dead)):
            # update q value with new state     
            Q_new = reward[idx]
            if not dead[idx]:
                # the maximum expected future reward
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # fitting temporal difference
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()



