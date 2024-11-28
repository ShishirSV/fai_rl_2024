import numpy as np
import random
from gymnasium.spaces import Box
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import Gen8EnvSinglePlayer
from poke_env.data import GenData

class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def _handle_battle_message(self, split_message):
        """Override to handle 'sentchoice' messages at the player level"""
        if len(split_message) > 1 and split_message[1] == 'sentchoice':
            return
        return super()._handle_battle_message(split_message)

    def create_battle(self):
        """Create a battle instance"""
        return AbstractBattle(
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

    def describe_embedding(self) -> Box:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )