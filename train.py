import asyncio
import numpy as np
from datetime import datetime
import os
import json
from tabulate import tabulate
from tqdm import tqdm
import matplotlib.pyplot as plt

from poke_env.player import (
    RandomPlayer,
    MaxBasePowerPlayer,
    SimpleHeuristicsPlayer,
    background_cross_evaluate,
)
from gymnasium.utils.env_checker import check_env

from environment import ImprovedRLPlayer
from agent import DQNAgent

class TrainingMetrics:
    def __init__(self):
        self.rewards = []
        self.wins = []
        self.losses = []
        self.win_rates = []
        self.avg_rewards = []
        self.epsilons = []
        
    def update(self, reward, win, epsilon):
        self.rewards.append(reward)
        self.wins.append(1 if win else 0)
        self.losses.append(0 if win else 1)
        
        # Calculate running averages
        window_size = 100
        self.win_rates.append(np.mean(self.wins[-window_size:]))
        self.avg_rewards.append(np.mean(self.rewards[-window_size:]))
        self.epsilons.append(epsilon)
    
    def plot(self, save_path):
        plt.figure(figsize=(15, 10))
        
        # Plot win rate
        plt.subplot(3, 1, 1)
        plt.plot(self.win_rates)
        plt.title('Win Rate (100-battle moving average)')
        plt.xlabel('Battle')
        plt.ylabel('Win Rate')
        
        # Plot average reward
        plt.subplot(3, 1, 2)
        plt.plot(self.avg_rewards)
        plt.title('Average Reward (100-battle moving average)')
        plt.xlabel('Battle')
        plt.ylabel('Reward')
        
        # Plot epsilon
        plt.subplot(3, 1, 3)
        plt.plot(self.epsilons)
        plt.title('Epsilon Value')
        plt.xlabel('Battle')
        plt.ylabel('Epsilon')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

async def train_dqn(env, agent, n_battles, metrics):
    total_rewards = 0
    
    for battle in tqdm(range(n_battles), desc="Training Progress"):
        state = env.reset()[0]
        done = False
        battle_reward = 0
        
        while not done:
            # Get action from agent
            action = agent.get_action(state)
            
            # Execute action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience in memory
            agent.memory.push(state, action, reward, next_state, done)
            
            # Train agent
            agent.train_step()
            
            state = next_state
            battle_reward += reward
            
        # Update metrics
        total_rewards += battle_reward
        metrics.update(
            reward=battle_reward,
            win=env.won_last_battle(),
            epsilon=agent.epsilon
        )
        
        # Save progress every 1000 battles
        if battle % 1000 == 0:
            agent.save(f'saved_models/checkpoint_{battle}')
            metrics.plot(f'training_plots/progress_{battle}.png')

async def evaluate_agent(agent, env, n_battles):
    wins = 0
    total_rewards = 0
    
    for _ in tqdm(range(n_battles), desc="Evaluation Progress"):
        state = env.reset()[0]
        done = False
        battle_reward = 0
        
        while not done:
            action = agent.get_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            battle_reward += reward
            
        if env.won_last_battle():
            wins += 1
        total_rewards += battle_reward
    
    return wins / n_battles, total_rewards / n_battles

async def main():
    # Create directories for saving results
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('training_plots', exist_ok=True)
    os.makedirs('evaluation_results', exist_ok=True)
    
    # Training parameters
    n_training_battles = 50000  # Increased training battles
    n_evaluation_battles = 1000
    
    # Initialize training environment and agent
    opponent = RandomPlayer(battle_format="gen8randombattle")
    train_env = ImprovedRLPlayer(
        battle_format="gen8randombattle",
        opponent=opponent,
        start_challenging=True
    )
    
    # Initialize agent
    state_shape = train_env.observation_space.shape
    n_actions = train_env.action_space.n
    agent = DQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_final=0.01,
        epsilon_decay=100000,
        memory_size=100000,
        batch_size=64,
        learning_rate=0.0001
    )
    
    # Initialize metrics
    metrics = TrainingMetrics()
    
    # Training phase
    print("Starting training...")
    start_time = datetime.now()
    await train_dqn(train_env, agent, n_training_battles, metrics)
    training_time = datetime.now() - start_time
    
    # Save final model and plots
    agent.save('saved_models/final_model')
    metrics.plot('training_plots/final_progress.png')
    
    # Evaluation phase
    print("\nEvaluating against different opponents...")
    evaluation_results = {}
    
    # List of opponents to evaluate against
    opponents = {
        'random': RandomPlayer(battle_format="gen8randombattle"),
        'max_power': MaxBasePowerPlayer(battle_format="gen8randombattle"),
        'heuristic': SimpleHeuristicsPlayer(battle_format="gen8randombattle")
    }
    
    # Evaluate against each opponent
    for opponent_name, opponent in opponents.items():
        eval_env = ImprovedRLPlayer(
            battle_format="gen8randombattle",
            opponent=opponent,
            start_challenging=True
        )
        
        win_rate, avg_reward = await evaluate_agent(agent, eval_env, n_evaluation_battles)
        evaluation_results[opponent_name] = {
            'win_rate': win_rate,
            'avg_reward': avg_reward
        }
        eval_env.close()
    
    # Cross evaluation
    print("\nPerforming cross evaluation...")
    players = [train_env.agent] + list(opponents.values())
    cross_eval_task = background_cross_evaluate(players, n_evaluation_battles // 10)
    cross_evaluation = cross_eval_task.result()
    
    # Save results
    results = {
        'training_time': str(training_time),
        'n_training_battles': n_training_battles,
        'n_evaluation_battles': n_evaluation_battles,
        'evaluation_results': evaluation_results,
        'cross_evaluation': {str(k): {str(k2): v2 for k2, v2 in v.items()} 
                           for k, v in cross_evaluation.items()}
    }
    
    with open('evaluation_results/results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print results
    print("\nTraining completed in:", training_time)
    print("\nEvaluation Results:")
    for opponent_name, results in evaluation_results.items():
        print(f"\nAgainst {opponent_name}:")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Average Reward: {results['avg_reward']:.2f}")
    
    print("\nCross Evaluation Results:")
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])
    print(tabulate(table))
    
    # Cleanup
    train_env.close()

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()