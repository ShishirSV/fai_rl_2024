import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from gymnasium.spaces import Box, Space
import logging

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import (
    Gen8EnvSinglePlayer,
    RandomPlayer,
)
from poke_env.data import GenData

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
        
        # Networks
        self.policy_net = DQN(state_shape, n_actions).to(self.device)
        
    def get_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        """Calculate the reward for the current battle state"""
        return self.reward_computing_helper(
            current_battle,
            fainted_value=2.0,
            hp_value=1.0,
            victory_value=30.0
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

async def main():
    env_player = None
    opponent = None
    
    try:
        # Create opponent
        opponent = RandomPlayer(
            battle_format="gen8randombattle",
            server_configuration={
                "authenticate": False,
                "server_url": "sim.smogon.com:8000"
            }
        )
        
        # Create the RL player
        env_player = SimpleRLPlayer(
            battle_format="gen8randombattle",
            opponent=opponent,
            start_challenging=True,
            server_configuration={
                "authenticate": False,
                "server_url": "sim.smogon.com:8000"
            }
        )

        # Start both players
        await opponent.start_servers()
        await env_player.start_servers()

        # Wait for connection establishment
        await asyncio.sleep(2)

        # Initialize agent
        state_shape = env_player.observation_space.shape
        n_actions = env_player.action_space.n
        agent = DQNAgent(state_shape, n_actions)

        # Training loop
        logger.info("Starting training...")
        for _ in range(5):  # 5 battles for testing
            done = False
            state = env_player.reset()
            while not done:
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env_player.step(action)
                done = terminated or truncated
                
                if not done:
                    state = next_state

        logger.info("Training completed")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        # Cleanup
        if env_player is not None:
            await env_player.close()
        if opponent is not None:
            await opponent.close()

if __name__ == "__main__":
    asyncio.run(main())