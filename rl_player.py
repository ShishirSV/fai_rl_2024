import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from gymnasium.spaces import Box, Space
from gymnasium.utils.env_checker import check_env
from tabulate import tabulate

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import (
    Gen8EnvSinglePlayer,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    background_cross_evaluate,
    background_evaluate_player,
)

from poke_env.data import GenData
from poke_env.environment import PokemonType

class CustomBattle(AbstractBattle):
    def parse_message(self, split_message):
        """Override parse_message to handle 'sentchoice' messages"""
        if len(split_message) > 1 and split_message[1] == 'sentchoice':
            # Ignore sentchoice messages as they don't affect the battle state
            return
        return super().parse_message(split_message)

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

# class ReplayMemory:
#     def __init__(self, capacity):
#         self.memory = deque(maxlen=capacity)
    
#     def push(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))
    
#     def sample(self, batch_size):
#         batch = random.sample(self.memory, batch_size)
#         state, action, reward, next_state, done = zip(*batch)
#         return (
#             torch.FloatTensor(state),
#             torch.LongTensor(action),
#             torch.FloatTensor(reward),
#             torch.FloatTensor(next_state),
#             torch.FloatTensor(done)
#         )
    
#     def __len__(self):
#         return len(self.memory)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        # Convert batch of tuples to tuple of arrays
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays first
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Convert numpy arrays to tensors
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
        gamma=0.5,
        epsilon_start=1.0,
        epsilon_final=0.05,
        epsilon_decay=10000,
        memory_size=10000,
        batch_size=32,
        learning_rate=0.00025,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.steps = 0
        
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

class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def create_battle(self):
        """Override to use CustomBattle instead of the default battle class"""
        return CustomBattle(
            battle_tag=f"battle_{random.randrange(0x100000000):08x}",
            username=self.username,
            logger=self.logger,
        )
    
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle):
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GenData.from_gen(8).type_chart
                )

        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

async def train_dqn(env, agent, n_steps):
    state = env.reset()[0]  # Get initial state from the tuple
    for step in range(n_steps):
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)  # Unpack all 5 values
        done = terminated or truncated
        
        agent.memory.push(state, action, reward, next_state, done)
        agent.train_step()
        
        if done:
            state = env.reset()[0]  # Get initial state from the tuple
        else:
            state = next_state

async def test_dqn(env, agent, n_episodes):
    for episode in range(n_episodes):
        state = env.reset()[0]  # Get initial state from the tuple
        done = False
        while not done:
            action = agent.get_action(state, training=False)
            state, _, terminated, truncated, _ = env.step(action)  # Unpack all 5 values
            done = terminated or truncated

async def main():
    # Test environment
    opponent = RandomPlayer(battle_format="gen8randombattle")
    test_env = SimpleRLPlayer(
        battle_format="gen8randombattle", start_challenging=True, opponent=opponent
    )
    check_env(test_env)
    test_env.close()

    # Create training and evaluation environments
    opponent = RandomPlayer(battle_format="gen8randombattle")
    train_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )
    opponent = RandomPlayer(battle_format="gen8randombattle")
    eval_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )

    # Initialize agent
    state_shape = train_env.observation_space.shape
    n_actions = train_env.action_space.n
    agent = DQNAgent(state_shape, n_actions)

    # Training
    await train_dqn(train_env, agent, n_steps=10)
    train_env.close()

    # Evaluation
    print("Results against random player:")
    await test_dqn(eval_env, agent, n_episodes=1)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )

    # Test against max base power player
    second_opponent = MaxBasePowerPlayer(battle_format="gen8randombattle")
    eval_env.reset_env(restart=True, opponent=second_opponent)
    print("Results against max base power player:")
    await test_dqn(eval_env, agent, n_episodes=100)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    eval_env.reset_env(restart=False)

    # Evaluate using included util method
    n_challenges = 25
    placement_battles = 4
    eval_task = background_evaluate_player(
        eval_env.agent, n_challenges, placement_battles
    )
    await test_dqn(eval_env, agent, n_episodes=n_challenges)
    print("Evaluation with included method:", eval_task.result())
    eval_env.reset_env(restart=False)

    # Cross evaluation
    n_challenges = 5
    players = [
        eval_env.agent,
        RandomPlayer(battle_format="gen8randombattle"),
        MaxBasePowerPlayer(battle_format="gen8randombattle"),
        SimpleHeuristicsPlayer(battle_format="gen8randombattle"),
    ]
    cross_eval_task = background_cross_evaluate(players, n_challenges)
    await test_dqn(eval_env, agent, n_episodes=n_challenges * (len(players) - 1))
    cross_evaluation = cross_eval_task.result()
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])
    print("Cross evaluation of DQN with baselines:")
    print(tabulate(table))
    eval_env.close()

if __name__ == "__main__":
    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()