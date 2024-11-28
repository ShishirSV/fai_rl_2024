import asyncio
from poke_env.player import (
    RandomPlayer,
    MaxBasePowerPlayer,
    SimpleHeuristicsPlayer,
    background_cross_evaluate,
    background_evaluate_player,
)
from tabulate import tabulate
from gymnasium.utils.env_checker import check_env
from environment import SimpleRLPlayer
from agent import DQNAgent
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

async def train_dqn(env, agent, n_steps):
    state = env.reset()[0]
    losses = []
    rewards = []
    episode_reward = 0
    
    for step in range(n_steps):
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        agent.memory.push(state, action, reward, next_state, done)
        loss = agent.train_step()
        
        if loss is not None:
            losses.append(loss)
        
        if done:
            rewards.append(episode_reward)
            episode_reward = 0
            state = env.reset()[0]
        else:
            state = next_state
            
        if step % 10 == 0:
            print(f"Step {step}/{n_steps} | "
                  f"Avg Loss: {np.mean(losses[-100:]) if losses else 0:.4f} | "
                  f"Avg Reward: {np.mean(rewards[-100:]) if rewards else 0:.2f}")
    
    return losses, rewards

async def test_dqn(env, agent, n_episodes):
    test_rewards = []
    win_rates = []
    
    for episode in range(n_episodes):
        state = env.reset()[0]
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        test_rewards.append(episode_reward)
        win_rate = env.n_won_battles / env.n_finished_battles
        win_rates.append(win_rate)
        
        if (episode + 1) % 10 == 0:
            print(f"Test Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {np.mean(test_rewards[-10:]):.2f} | "
                  f"Win Rate: {win_rate:.2f}")
    
    return test_rewards, win_rates

def plot_metrics(train_losses, train_rewards, test_rewards, win_rates, opponent_name):
    os.makedirs('training_plots', exist_ok=True)
    
    # Plot training losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f'training_plots/training_loss.png')
    plt.close()
    
    # Plot training rewards
    plt.figure(figsize=(10, 6))
    plt.plot(train_rewards)
    plt.title('Training Rewards Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.grid(True)
    plt.savefig(f'training_plots/training_rewards.png')
    plt.close()
    
    # Plot test rewards
    plt.figure(figsize=(10, 6))
    plt.plot(test_rewards)
    plt.title(f'Test Rewards Against {opponent_name}')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.grid(True)
    plt.savefig(f'training_plots/test_rewards_{opponent_name}.png')
    plt.close()
    
    # Plot win rates
    plt.figure(figsize=(10, 6))
    plt.plot(win_rates)
    plt.title(f'Win Rate Against {opponent_name}')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.grid(True)
    plt.savefig(f'training_plots/win_rate_{opponent_name}.png')
    plt.close()

async def main():
    # Test environment setup
    opponent = RandomPlayer(battle_format="gen8randombattle")
    test_env = SimpleRLPlayer(
        battle_format="gen8randombattle", start_challenging=True, opponent=opponent
    )
    check_env(test_env)
    test_env.close()

    # Create environments
    opponent = RandomPlayer(battle_format="gen8randombattle")
    train_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )
    opponent = RandomPlayer(battle_format="gen8randombattle")
    eval_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )

    # Initialize agent
    state_shape = train_env.observation_space.shape
    n_actions = train_env.action_space.n
    agent = DQNAgent(state_shape, n_actions)

    # Training
    print("Training DQN agent...")
    train_losses, train_rewards = await train_dqn(train_env, agent, n_steps=10000)
    
    # Save model
    print("Saving trained model...")
    agent.save()
    train_env.close()

    # Evaluation against random player
    print("Results against random player:")
    test_rewards_random, win_rates_random = await test_dqn(eval_env, agent, n_episodes=100)
    print(f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes")
    
    # Plot metrics for random player
    plot_metrics(train_losses, train_rewards, test_rewards_random, win_rates_random, "random_player")

    # Test against max base power player
    second_opponent = MaxBasePowerPlayer(battle_format="gen8randombattle")
    eval_env.reset_env(restart=True, opponent=second_opponent)
    print("Results against max base power player:")
    test_rewards_max, win_rates_max = await test_dqn(eval_env, agent, n_episodes=100)
    print(f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes")
    
    # Plot metrics for max base power player
    plot_metrics(train_losses, train_rewards, test_rewards_max, win_rates_max, "max_power_player")
    
    eval_env.reset_env(restart=False)

    # Cross evaluation
    n_challenges = 5
    players = [
        eval_env.agent,
        RandomPlayer(battle_format="gen8randombattle"),
        MaxBasePowerPlayer(battle_format="gen8randombattle"),
        SimpleHeuristicsPlayer(battle_format="gen8randombattle"),
    ]
    cross_eval_task = background_cross_evaluate(players, n_challenges)
    await test_dqn(eval_env, agent, n_episodes=n_challenges * (len(players) - 1))
    cross_evaluation = cross_eval_task.result()
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])
    print("Cross evaluation of DQN with baselines:")
    print(tabulate(table))
    eval_env.close()

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()