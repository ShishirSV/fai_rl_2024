from poke_env.environment.abstract_battle import AbstractBattle

class CustomBattle(AbstractBattle):
    def parse_message(self, split_message):
        """Override to handle additional message types"""
        if len(split_message) > 1 and split_message[1] == 'sentchoice':
            # Ignore sentchoice messages as they don't affect battle state
            return
        return super().parse_message(split_message)