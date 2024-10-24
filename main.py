import asyncio
from environments.pokemon_env import PokemonRLEnv
from agents.dqn_agent import DQNAgent
from poke_env.player.random_player import RandomPlayer

async def train(env, agent, episodes: int = 1000):
    opponent = RandomPlayer(battle_format="gen8randombattle")
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        await env.battle_against(opponent=opponent, n_battles=1)
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.memory.push(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            total_reward += reward
        
        print(f"Episode {episode + 1}/{episodes}")
        print(f"Total reward: {total_reward}")
        print(f"Epsilon: {agent.epsilon}")
        print(f"Win rate: {env.n_won_battles/(env.n_won_battles + env.n_lost_battles):.2%}")

if __name__ == "__main__":
    env = PokemonRLEnv(username="RL_Bot")
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n
    )
    
    asyncio.get_event_loop().run_until_complete(train(env, agent))