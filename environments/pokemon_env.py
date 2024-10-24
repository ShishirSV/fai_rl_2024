from poke_env import AccountConfiguration
from poke_env.player.player import Player
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from gym.spaces import Discrete, Box
import numpy as np
from typing import Optional, Tuple, Dict

class PokemonRLEnv(Player):
    def __init__(
        self,
        username: str = "RL_Bot",
        password: Optional[str] = None,
        battle_format: str = "gen8randombattle"
    ):
        # Setup account configuration
        account_config = AccountConfiguration(username, password)
        
        # Initialize the parent class (Player)
        super().__init__(
            battle_format=battle_format,
            account_configuration=account_config
        )
        
        # Define spaces
        self.action_space = Discrete(9)  # 4 moves + 5 switches
        self.observation_space = Box(
            low=0,
            high=1,
            shape=(128,),
            dtype=np.float32
        )
        
        self.current_battle = None
        self.last_reward = 0
        self._reward_range = (-1, 1)

    def compute_reward(self, battle: AbstractBattle) -> float:
        reward = 0
        
        # Reward for fainting opponent's Pokemon
        if battle.opponent_active_pokemon.fainted:
            reward += 1.0
            
        # Penalty for having our Pokemon faint
        if battle.active_pokemon.fainted:
            reward -= 1.0
            
        # Reward based on HP difference
        my_hp_fraction = battle.active_pokemon.current_hp / battle.active_pokemon.max_hp
        opponent_hp_fraction = battle.opponent_active_pokemon.current_hp / battle.opponent_active_pokemon.max_hp
        reward += 0.5 * (my_hp_fraction - opponent_hp_fraction)
        
        if battle.won:
            reward += 2.0
        elif battle.lost:
            reward -= 2.0
            
        return reward

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        state = np.zeros(128)
        current_index = 0
        
        # Team HP
        for pokemon in battle.team.values():
            hp_fraction = pokemon.current_hp / pokemon.max_hp if pokemon.current_hp is not None else 0
            state[current_index] = hp_fraction
            current_index += 1
            
        # Opponent HP
        for pokemon in battle.opponent_team.values():
            hp_fraction = pokemon.current_hp / pokemon.max_hp if pokemon.current_hp is not None else 0
            state[current_index] = hp_fraction
            current_index += 1
        
        return state

    def reset(self) -> np.ndarray:
        self.current_battle = None
        self.last_reward = 0
        return np.zeros(self.observation_space.shape)
    
    def choose_move(self, battle: AbstractBattle, order: BattleOrder) -> BattleOrder:
        # Here we implement a simple logic to choose a move or switch based on the order
        if battle.available_moves and order.move in battle.available_moves:
            return order
        elif battle.available_switches and order.pokemon in battle.available_switches:
            return order
        else:
            # Fallback to the first available move if the selected move is invalid
            return BattleOrder(battle.available_moves[0] if battle.available_moves else None)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if action < 4:  # Moves
            battle_move = self.current_battle.available_moves[action]
            order = BattleOrder(battle_move)
        else:  # Switches
            switch_idx = action - 4
            if switch_idx < len(self.current_battle.available_switches):
                pokemon = self.current_battle.available_switches[switch_idx]
                order = BattleOrder(pokemon)
            else:
                order = BattleOrder(self.current_battle.available_moves[0])
        
        self.current_battle = self.choose_move(self.current_battle, order)
        
        state = self.embed_battle(self.current_battle)
        reward = self.compute_reward(self.current_battle)
        done = self.current_battle.finished
        
        info = {
            'won': self.current_battle.won,
            'lost': self.current_battle.lost,
            'invalid_action': action >= 4 + len(self.current_battle.available_switches)
        }
        
        return state, reward, done, info