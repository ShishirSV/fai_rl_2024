import random
import torch
import torch.optim as optim
import torch.nn as nn
import os
from models import DQN
from memory import ReplayMemory

class DQNAgent:
    def __init__(
        self,
        state_shape,
        n_actions,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_final=0.01,
        epsilon_decay=50000,
        memory_size=100000,
        batch_size=32,
        learning_rate=0.00001,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.steps = 0
        self.state_shape = state_shape  # Store for model loading
        
        # Networks
        self.policy_net = DQN(state_shape, n_actions).to(self.device)
        self.target_net = DQN(state_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.memory = ReplayMemory(memory_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
    
    def get_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def update_epsilon(self):
        self.epsilon = max(
            self.epsilon_final,
            self.epsilon - (1.0 - self.epsilon_final) / self.epsilon_decay
        )
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        if self.steps % 1000 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.steps += 1
        self.update_epsilon()

    def save(self, path='saved_models'):
        """Save the trained model and training state"""
        os.makedirs(path, exist_ok=True)
        
        # Save model parameters and training state
        state = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'state_shape': self.state_shape,
            'n_actions': self.n_actions
        }
        torch.save(state, os.path.join(path, 'dqn_model.pth'))
    
    @classmethod
    def load(cls, path='saved_models/dqn_model.pth'):
        """Load a trained model and return a new agent instance"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model found at {path}")
            
        # Load the saved state
        state = torch.load(path)
        
        # Create a new agent instance
        agent = cls(
            state_shape=state['state_shape'],
            n_actions=state['n_actions']
        )
        
        # Load the saved parameters
        agent.policy_net.load_state_dict(state['policy_net_state_dict'])
        agent.target_net.load_state_dict(state['target_net_state_dict'])
        agent.optimizer.load_state_dict(state['optimizer_state_dict'])
        agent.epsilon = state['epsilon']
        agent.steps = state['steps']
        
        return agent