from poke_env import RandomPlayer
from poke_env.data import GenData
from poke_env import AccountConfiguration
from poke_env import cross_evaluate
from gymnasium.utils.env_checker import check_env
from agent import DQNAgent
from environment import SimpleRLPlayer
import asyncio
import os

async def main():
    # Initialize environment and opponent
    opponent = RandomPlayer(battle_format="gen8randombattle")
    env = SimpleRLPlayer(
        battle_format="gen8randombattle",
        opponent=opponent,
        start_challenging=True
    )
    
    # Initialize or load the agent
    model_path = 'saved_models/dqn_model_final.pth'
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        agent = DQNAgent.load(model_path)
    else:
        print("No pre-trained model found. Initializing new agent...")
        state_shape = env.observation_space.shape
        n_actions = env.action_space.n
        agent = DQNAgent(state_shape, n_actions)
    
    # Run a test battle
    n_battles = 1
    total_rewards = 0
    
    for _ in range(n_battles):
        state = env.reset()[0]
        done = False
        while not done:
            action = agent.get_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_rewards += reward
    
    print(f"Average reward over {n_battles} battles: {total_rewards/n_battles}")
    print(f"Won {env.n_won_battles} out of {env.n_finished_battles} battles")
    
    env.close()

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()