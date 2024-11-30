import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from gymnasium.spaces import Box
from gymnasium.utils.env_checker import check_env
from poke_env.player import RandomPlayer
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env import AccountConfiguration
from poke_env.player.player import Player
from typing import List, Optional
from tabulate import tabulate

class CustomBattle(AbstractBattle):
    def parse_message(self, split_message):
        """Override parse_message to handle 'sentchoice' messages"""
        if len(split_message) > 1 and split_message[1] == 'sentchoice':
            return None
        return super().parse_message(split_message)

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(
        self,
        state_shape,
        n_actions,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_final=0.05,
        epsilon_decay=10000,
        memory_size=10000,
        batch_size=32,
        learning_rate=0.00025,
        target_update_freq=1000
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps = 0
        
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
        
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.steps += 1
        self.update_epsilon()

class SimpleRLPlayer(Player):
    def __init__(self, battle_format: str, opponent: Optional[Player] = None):
        super().__init__(battle_format=battle_format)
        self.state_shape = (10,)  # Based on our embedding size
        self.n_actions = 4  # Number of possible moves
        self.agent = DQNAgent(self.state_shape, self.n_actions)
        self.opponent = opponent
        
    def create_battle(self):
        return CustomBattle(
            battle_tag=f"battle_{random.randrange(0x100000000):08x}",
            username=self.username,
            logger=self.logger,
        )
    
    def choose_move(self, battle):
        state = self.embed_battle(battle)
        action = self.agent.get_action(state)
        
        # Convert action index to move
        available_moves = battle.available_moves
        if action < len(available_moves):
            return available_moves[action]
        else:
            # Fallback to first available move if action is invalid
            return available_moves[0]
    
    def embed_battle(self, battle):
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (move.base_power or 0) / 100
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2
                )

        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6

        return np.float32(np.concatenate([
            moves_base_power,
            moves_dmg_multiplier,
            [fainted_mon_team, fainted_mon_opponent]
        ]))

    def describe_embedding(self) -> Box:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32
        )

async def train_agent(player: SimpleRLPlayer, n_battles: int):
    # Training loop
    for _ in range(n_battles):
        battle = player.create_battle()
        await battle.set_opponent(player.opponent)
        
        state = player.embed_battle(battle)
        done = False
        
        while not done:
            action = player.agent.get_action(state, training=True)
            move = player.choose_move(battle)
            await battle.make_move(move)
            
            next_state = player.embed_battle(battle)
            reward = battle.won  # Simple reward: 1 for win, 0 for ongoing/loss
            done = battle.finished
            
            player.agent.memory.push(state, action, reward, next_state, done)
            player.agent.train_step()
            
            state = next_state

async def evaluate_agent(player: SimpleRLPlayer, n_battles: int):
    wins = 0
    for _ in range(n_battles):
        battle = player.create_battle()
        await battle.set_opponent(player.opponent)
        
        while not battle.finished:
            move = player.choose_move(battle)
            await battle.make_move(move)
        
        if battle.won:
            wins += 1
    
    return wins / n_battles

async def main():
    # Create players
    my_account_config = AccountConfiguration("MyTrainerName", None)
    
    # Initialize opponent
    random_player = RandomPlayer(
        battle_format="gen8randombattle",
        account_configuration=my_account_config
    )
    
    # Initialize our RL player
    rl_player = SimpleRLPlayer(
        battle_format="gen8randombattle",
        opponent=random_player
    )
    
    # Training phase
    print("Starting training...")
    await train_agent(rl_player, n_battles=100)
    print("Training completed")
    
    # Evaluation phase
    print("\nStarting evaluation...")
    win_rate = await evaluate_agent(rl_player, n_battles=50)
    print(f"Win rate against random player: {win_rate:.2%}")

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())