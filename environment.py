import numpy as np
from gymnasium.spaces import Box
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import Gen8EnvSinglePlayer
from poke_env.data import GenData
from custom_battle import CustomBattle
import random

class ImprovedRLPlayer(Gen8EnvSinglePlayer):
    def create_battle(self):
        """Override to use CustomBattle instead of the default battle class"""
        return CustomBattle(
            battle_tag=f"battle_{random.randrange(0x100000000):08x}",
            username=self.username,
            logger=self.logger,
        )

    def calc_reward(self, last_battle, current_battle) -> float:
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

        # Pokemon HP
        active_pokemon_hp = battle.active_pokemon.current_hp_fraction
        opponent_hp = battle.opponent_active_pokemon.current_hp_fraction
        
        # Status conditions
        status_conditions = np.zeros(6)
        if battle.opponent_active_pokemon.status:
            status_mapping = {
                'slp': 0, 'psn': 1, 'brn': 2,
                'frz': 3, 'par': 4, 'tox': 5
            }
            status = battle.opponent_active_pokemon.status.value
            if status in status_mapping:
                status_conditions[status_mapping[status]] = 1
        
        # Team statistics
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        
        final_vector = np.concatenate([
            moves_base_power,
            moves_dmg_multiplier,
            [active_pokemon_hp, opponent_hp],
            status_conditions,
            [fainted_mon_team, fainted_mon_opponent],
        ])
        
        return np.float32(final_vector)

    def describe_embedding(self) -> Box:
        n_features = 4 + 4 + 2 + 6 + 2  # Moves power, multiplier, HPs, status, fainted
        return Box(
            low=-1.0,
            high=4.0,
            shape=(n_features,),
            dtype=np.float32
        )