import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_shape[0], 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, n_actions)
        )
    
    def forward(self, x):
        return self.network(x)