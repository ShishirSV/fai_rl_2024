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

async def train_dqn(env, agent, n_steps):
    state = env.reset()[0]
    for step in range(n_steps):
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        agent.memory.push(state, action, reward, next_state, done)
        agent.train_step()
        
        if done:
            state = env.reset()[0]
        else:
            state = next_state

async def test_dqn(env, agent, n_episodes):
    for episode in range(n_episodes):
        state = env.reset()[0]
        done = False
        while not done:
            action = agent.get_action(state, training=False)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

async def main():
    # Test environment
    opponent = RandomPlayer(battle_format="gen8randombattle")
    test_env = SimpleRLPlayer(
        battle_format="gen8randombattle", start_challenging=True, opponent=opponent
    )
    check_env(test_env)
    test_env.close()

    # Create training and evaluation environments
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
    await train_dqn(train_env, agent, n_steps=100000)

    # Save the trained model
    print("Saving trained model...")
    agent.save()

    train_env.close()

    # Evaluation
    print("Results against random player:")
    await test_dqn(eval_env, agent, n_episodes=100)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )

    # Test against max base power player
    second_opponent = MaxBasePowerPlayer(battle_format="gen8randombattle")
    eval_env.reset_env(restart=True, opponent=second_opponent)
    print("Results against max base power player:")
    await test_dqn(eval_env, agent, n_episodes=100)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
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