# Pokemon Battle AI using Deep Q-Learning

A reinforcement learning agent that learns to play Pokemon battles using Deep Q-Learning (DQN). The agent learns optimal battle strategies by playing against various opponent types.

## Features

- Deep Q-Learning implementation for Pokemon battle decision making
- Custom Pokemon battle environment using poke-env
- Battle visualization and real-time statistics
- Performance evaluation against multiple baseline strategies
- Interactive mode to play against the trained model
- Training metrics visualization

## Requirements

```
python >= 3.8
torch
numpy
matplotlib
poke-env
gymnasium
tabulate
```

## Training

To train the model:

```bash
python train.py
```

The script will:
- Train the agent against random opponents
- Generate performance plots
- Save the trained model
- Evaluate against different opponent types


## Model Architecture

- Dueling DQN with experience replay
- State representation: Pokemon stats, moves, battle conditions
- Action space: Available moves and switches
- Reward structure: Victory/defeat, HP changes, strategic actions

## Performance

- Training metrics stored in `training_plots/`
- Cross-evaluation against baseline agents:
  - Random Player
  - Max Base Power Player
  - Simple Heuristics Player
