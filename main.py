from poke_env import RandomPlayer
from poke_env.data import GenData
from poke_env import AccountConfiguration
from poke_env import cross_evaluate
import max_damage as md
from gymnasium.utils.env_checker import check_env
from rl_player import SimpleRLPlayer

# The RandomPlayer is a basic agent that makes decisions randomly,
# serving as a starting point for more complex agent development.

# The battle_against method initiates a battle between two players.
# Here we are using asynchronous programming (await) to start the battle.
import asyncio

async def main():
    my_account_config1 = AccountConfiguration("my_username_123", None)
    # my_account_config2 = AccountConfiguration("my_username_12345", None)
    
    random_player = RandomPlayer(battle_format="gen8randombattle")
    rl_player = SimpleRLPlayer(battle_format="gen8randombattle", opponent=random_player)


    await random_player.battle_against(rl_player, n_battles=1)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())