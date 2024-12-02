import asyncio
from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from environment import SimpleRLPlayer
from agent import DQNAgent

async def test_battles(agent, opponent, n_battles=1000):
    env = SimpleRLPlayer(battle_format="gen8randombattle", opponent=opponent, start_challenging=True)
    
    for _ in range(n_battles):
        state = env.reset()[0]
        done = False
        while not done:
            action = agent.get_action(state, training=False)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
    win_rate = env.n_won_battles / env.n_finished_battles
    print(f"Won {env.n_won_battles} / {env.n_finished_battles} battles (Win rate: {win_rate:.2%})")
    env.close()
    return win_rate

async def main():
    agent = DQNAgent.load('saved_models/dqn_model_final.pth')
    
    opponents = {
        "Random": RandomPlayer(battle_format="gen8randombattle"),
        "MaxPower": MaxBasePowerPlayer(battle_format="gen8randombattle"),
        "Heuristic": SimpleHeuristicsPlayer(battle_format="gen8randombattle")
    }
    
    for name, opponent in opponents.items():
        print(f"\nTesting against {name} player:")
        await test_battles(agent, opponent)

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()