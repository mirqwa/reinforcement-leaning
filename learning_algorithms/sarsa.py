import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.abspath('../reinforcement-leaning'))

import actions
import environment
import plot
import qtable
import utils


def sarsa(sim_input, sim_output) -> (np.array, list):
    """
    SARSA: on-policy RL algorithm to train agent
    """
    num_episodes = sim_input.num_episodes
    gamma = sim_input.gamma
    alpha = sim_input.alpha
    epsilon = sim_input.epsilon

    q_table = qtable.init_q_table()
    steps_cache = np.zeros(num_episodes)
    rewards_cache = np.zeros(num_episodes)

    # Iterate over episodes
    for episode in range(num_episodes):

        # Set to target policy at final episode
        if episode == len(range(num_episodes)) - 1:
            epsilon = 0

        # Initialize environment and agent position
        agent_pos, env, cliff_pos, goal_pos, game_over = environment.init_env()

        while not game_over:

            if steps_cache[episode] == 0:
                # Get state corresponding to agent position
                state = environment.get_state(agent_pos)

                # Select action using ε-greedy policy
                action = actions.epsilon_greedy_action(state, q_table, epsilon)

            # Move agent to next position
            agent_pos = actions.move_agent(agent_pos, action)

            # Mark visited path
            env = environment.mark_path(agent_pos, env)

            # Determine next state
            next_state = environment.get_state(agent_pos)

            # Compute and store reward
            reward = actions.get_reward(next_state, cliff_pos, goal_pos)
            rewards_cache[episode] += reward

            # Check whether game is over
            game_over = environment.check_game_over(episode,
                next_state, cliff_pos, goal_pos, steps_cache[episode]
            )

            # Select next action using ε-greedy policy
            next_action = actions.epsilon_greedy_action(next_state, q_table, epsilon)

            # Determine Q-value next state (on-policy)
            next_state_value = q_table[next_action][next_state]

            # Update Q-table
            q_table = qtable.update_q_table(
                q_table, state, action, reward, next_state_value, gamma, alpha
            )

            # Update state and action
            state = next_state
            action = next_action

            steps_cache[episode] += 1

    sim_output.step_cache.append(steps_cache)
    sim_output.reward_cache.append(rewards_cache)

    sim_output.env_cache.append(env)  # array of np arrays
    sim_output.name_cache.append("SARSA")

    return q_table, sim_output


def plot_simulation_results(sim_input, sim_output):
    plot.console_output(
        sim_output,
        sim_input.num_episodes,
    )
    # Plot output
    plot.plot_steps(sim_output)
    plot.plot_rewards(sim_output)
    plot.plot_path(sim_output)

def main(num_episodes, gamma, alpha, epsilon):
    sim_input = utils.sim_init(num_episodes=num_episodes, gamma=gamma, alpha=alpha, epsilon=epsilon)
    sim_output = utils.sim_output(
        rewards_cache=[], step_cache=[], env_cache=[], name_cache=[]
    )
    q_table_sarsa, sim_output = sarsa(sim_input, sim_output)
    plot_simulation_results(sim_input, sim_output)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--num_episodes", default=10000, type=int)
    args.add_argument("--gamma", default=0.8, type=float)
    args.add_argument("--alpha", default=0.01, type=float)
    args.add_argument("--epsilon", default=0.1, type=float)
    args = args.parse_args()
    main(args.num_episodes, args.gamma, args.alpha, args.epsilon)