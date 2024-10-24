from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from poke_env import ServerConfiguration
import asyncio
from typing import Optional, Dict, Any
import logging

class ShowdownManager:
    """
    Manages Pokemon Showdown battles and server connections using poke-env.
    Coordinates battles between players and handles server configuration.
    """
    def __init__(
        self,
        server_url: str = "play.pokemonshowdown.com",
        auth_url: str = "https://play.pokemonshowdown.com/action.php",
        rate_limit: Optional[float] = 0.0
    ):
        self.server_configuration = ServerConfiguration(
            server_url=server_url,
            authentication_url=auth_url,
            rate_limit=rate_limit
        )
        self.logger = logging.getLogger(__name__)

    async def setup_player(
        self,
        player: Player,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Setup a player's connection to the server with retry logic.
        
        Args:
            player (Player): Player to connect
            max_retries (int): Maximum number of connection attempts
            retry_delay (float): Delay between retries in seconds
        """
        for attempt in range(max_retries):
            try:
                await player.connect_to_server(
                    server_configuration=self.server_configuration
                )
                self.logger.info(f"Successfully connected player: {player.username}")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(
                        f"Connection attempt {attempt + 1} failed for {player.username}. "
                        f"Retrying in {retry_delay} seconds..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    self.logger.error(f"Failed to connect player after {max_retries} attempts")
                    raise

    async def run_battles(
        self,
        player1: Player,
        player2: Player,
        n_battles: int = 1,
        battle_format: str = "gen8randombattle"
    ) -> Dict[str, Any]:
        """
        Run battles between two players.
        
        Args:
            player1 (Player): First player
            player2 (Player): Second player
            n_battles (int): Number of battles to run
            battle_format (str): Format for the battles
            
        Returns:
            Dict[str, Any]: Battle results statistics
        """
        try:
            # Ensure both players are connected
            await asyncio.gather(
                self.setup_player(player1),
                self.setup_player(player2)
            )
            
            # Start the battles
            await player1.battle_against(
                opponent=player2,
                n_battles=n_battles,
                battle_format=battle_format
            )
            
            # Calculate statistics
            results = {
                "player1_wins": player1.n_won_battles,
                "player2_wins": player2.n_won_battles,
                "total_battles": n_battles,
                "win_rate_player1": player1.n_won_battles / n_battles if n_battles > 0 else 0,
                "win_rate_player2": player2.n_won_battles / n_battles if n_battles > 0 else 0,
                "player1_username": player1.username,
                "player2_username": player2.username
            }
            
            self.logger.info(
                f"Completed {n_battles} battles between "
                f"{player1.username} and {player2.username}"
            )
            return results
            
        except Exception as e:
            self.logger.error(f"Battle execution failed: {str(e)}")
            raise

    async def run_ladder_battles(
        self,
        player: Player,
        n_battles: int = 1,
        battle_format: str = "gen8randombattle",
        team: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run ladder battles for a player.
        
        Args:
            player (Player): Player to ladder
            n_battles (int): Number of battles
            battle_format (str): Format for the battles
            team (Optional[str]): Team to use in battles
            
        Returns:
            Dict[str, Any]: Ladder results
        """
        try:
            await self.setup_player(player)
            
            if team:
                player.team = team
            
            await player.ladder(n_battles=n_battles, battle_format=battle_format)
            
            results = {
                "username": player.username,
                "wins": player.n_won_battles,
                "losses": player.n_lost_battles,
                "total_battles": n_battles,
                "win_rate": player.n_won_battles / n_battles if n_battles > 0 else 0,
                "format": battle_format
            }
            
            self.logger.info(
                f"Completed {n_battles} ladder battles for {player.username}"
            )
            return results
            
        except Exception as e:
            self.logger.error(f"Ladder battles failed: {str(e)}")
            raise

    async def test_battle_against_random(
        self,
        player: Player,
        n_battles: int = 1,
        battle_format: str = "gen8randombattle"
    ) -> Dict[str, Any]:
        """
        Test a player against a random opponent.
        
        Args:
            player (Player): Player to test
            n_battles (int): Number of test battles
            battle_format (str): Format for the battles
            
        Returns:
            Dict[str, Any]: Test results
        """
        random_agent = RandomPlayer(battle_format=battle_format)
        return await self.run_battles(
            player1=player,
            player2=random_agent,
            n_battles=n_battles,
            battle_format=battle_format
        )

    def get_configuration(self) -> Dict[str, Any]:
        """Get current server configuration"""
        return {
            "server_url": self.server_configuration.server_url,
            "authentication_url": self.server_configuration.authentication_url,
            "rate_limit": self.server_configuration.rate_limit
        }