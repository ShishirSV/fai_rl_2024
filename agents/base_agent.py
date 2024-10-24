from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Tuple

class BaseAgent(ABC):
    """
    Abstract base class that defines the interface for all RL agents.
    Any new agent implementation should inherit from this class.
    """
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.training = True
    
    @abstractmethod
    def choose_action(self, state: np.ndarray) -> int:
        """
        Select an action based on the current state.
        
        Args:
            state (np.ndarray): Current state observation
            
        Returns:
            int: Selected action index
        """
        pass
    
    @abstractmethod
    def learn(self):
        """
        Update the agent's policy based on experiences.
        Should be implemented by specific agent types.
        """
        pass
    
    def save(self, path: str):
        """
        Save agent's model/parameters to disk.
        
        Args:
            path (str): Path to save the model
        """
        raise NotImplementedError
    
    def load(self, path: str):
        """
        Load agent's model/parameters from disk.
        
        Args:
            path (str): Path to load the model from
        """
        raise NotImplementedError
    
    def train(self):
        """Set agent to training mode"""
        self.training = True
    
    def eval(self):
        """Set agent to evaluation mode"""
        self.training = False
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> Dict[str, Any]:
        """
        Update the agent with a new experience tuple.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether episode is done
            
        Returns:
            Dict[str, Any]: Dictionary containing training metrics
        """
        if self.training:
            self.learn()
        return {}
    
    def reset(self):
        """Reset agent's episode-specific variables"""
        pass