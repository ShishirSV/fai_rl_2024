# from poke_env.server_configuration import ShowdownServerConfiguration
from poke_env.player.random_player import RandomPlayer

# showdown_config = ShowdownServerConfiguration("https://play.pokemonshowdown.com", "https://play.pokemonshowdown.com")
# from poke_env.player.random_player import RandomPlayer

random_player = RandomPlayer(
    battle_format="gen9randombattle",
    # server_url="https://play.pokemonshowdown.com",
    # server_port=None,  # Use default port
)

# random_player = RandomPlayer(battle_format="gen9randombattle", server_configuration=showdown_config)
