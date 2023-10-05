# COBRA :snake:

AI agent learns how to play Snake using Deep Reinforcement Learning (PyTorch).

![Alt Text](https://github.com/brightonm/COBRA/blob/main/pretrained_model.gif)

The goal of this project is to develop an AI Bot able to learn how to play the Snake game from scratch.

* Implemented the Snake game from scratch using pygame library
* Implemented a Deep Reinforcement Learning algorithm : Q-Learning and use a neural network to approximation the policy. Besides, I use epsilon-greedy exploration.
* Visualized  how the Deep Q-Learning algorithm learns how to play snake, scoring up to 67 points after only 10 minutes of training.

## Installation :construction_worker:

Install the following packages in your **pytorch environnement**:

```bash
pip install numpy
pip install pygame
```

The code was tested on Python 3.10 and PyTorch 2.0.

## How to play the game :video_game:

```bash
python play_cobra_game.py
```

## How to watch the agent train and learn :elephant:

```bash
python rl_agent.py
```

Models will be saved in the `models/` folder.

## How to a use pretrained model and watch its performance :rocket:

```bash
python rl_agent --model_path pretrained_models/pretrained_model.pth
```

## License
This code is distributed under an [MIT LICENSE](LICENSE).
