import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath("../reinforcement-leaning"))

import actions
import environment
import qtable
import utils


def state_action_exists_earlier(
    earlier_state_trajectory, earlier_action_trajectory, state, action
):
    states = np.array(earlier_state_trajectory)
    actions = np.array(earlier_action_trajectory)
    state_indices = list(np.where(states == state)[0])
    action_indices = list(np.where(actions == action)[0])
    common_indices = list(set(state_indices).intersection(action_indices))
    return common_indices


def update_q_table(
    reward_trajectory,
    action_trajectory,
    state_trajectory,
    gamma,
    q_table,
    alpha,
    first_visit,
):
    for t in range(len(reward_trajectory) - 1, -1, -1):
        reward = reward_trajectory[t]
        action = action_trajectory[t]
        state = state_trajectory[t]
        if first_visit and state_action_exists_earlier(
            state_trajectory[0:t], action_trajectory[0:t], state, action
        ):
            continue
        cum_reward = actions.compute_cum_rewards(gamma, t, reward_trajectory) + reward
        q_table[action, state] += alpha * (cum_reward - q_table[action, state])


def compute_and_store_reward(
    episode,
    state,
    next_state,
    action,
    cliff_pos,
    goal_pos,
    rewards_cache,
    state_trajectory,
    action_trajectory,
    reward_trajectory,
):
    # Compute and store reward
    reward = actions.get_reward(next_state, cliff_pos)
    rewards_cache[episode] += reward
    state_trajectory.append(state)
    action_trajectory.append(action)
    reward_trajectory.append(reward)


def update_simulation_output(env, steps_cache, rewards_cache, sim_output):
    sim_output.step_cache.append(steps_cache)
    sim_output.reward_cache.append(rewards_cache)
    sim_output.env_cache.append(env)
    sim_output.name_cache.append("Monte Carlo")


def monte_carlo(sim_input, sim_output, first_visit) -> (np.array, list):
    """
    Monte Carlo: full-trajectory RL algorithm to train agent
    """
    num_episodes = sim_input.num_episodes
    gamma = sim_input.gamma
    alpha = sim_input.alpha
    epsilon = sim_input.epsilon

    q_table = qtable.init_q_table()
    steps_cache = np.zeros(num_episodes)
    rewards_cache = np.zeros(num_episodes)

    # Generate the episodes
    for episode in range(num_episodes):
        # Set to target policy at final episode
        if episode == len(range(num_episodes)) - 1:
            epsilon = 0
        # Initialize environment and agent position for a new episode
        agent_pos, env, cliff_pos, goal_pos, game_over = environment.init_env()

        state_trajectory = []
        action_trajectory = []
        reward_trajectory = []

        state = environment.get_state(agent_pos)
        while not game_over:
            action = actions.epsilon_greedy_action(state, q_table, epsilon)
            # Move agent to next position
            agent_pos = actions.move_agent(agent_pos, action)
            # updating the environmet after taking the action in the current state
            env, next_state, game_over = environment.update_environment(
                env, episode, agent_pos, cliff_pos, goal_pos, steps_cache
            )
            compute_and_store_reward(
                episode,
                state,
                next_state,
                action,
                cliff_pos,
                goal_pos,
                rewards_cache,
                state_trajectory,
                action_trajectory,
                reward_trajectory,
            )
            state = next_state
            steps_cache[episode] += 1

        # At end of episode, update Q-table for full trajectory
        update_q_table(
            reward_trajectory,
            action_trajectory,
            state_trajectory,
            gamma,
            q_table,
            alpha,
            first_visit,
        )

    update_simulation_output(env, steps_cache, rewards_cache, sim_output)

    return q_table, sim_output


def main(num_episodes, gamma, alpha, epsilon, first_visit, plot_simulation=False):
    sim_input = utils.sim_init(
        num_episodes=num_episodes, gamma=gamma, alpha=alpha, epsilon=epsilon
    )
    sim_output = utils.sim_output(
        rewards_cache=[], step_cache=[], env_cache=[], name_cache=[]
    )
    q_table_mc, sim_output = monte_carlo(sim_input, sim_output, first_visit)
    q_table_df = pd.DataFrame(
        columns=["up", "down", "left", "right"], data=q_table_mc.T
    )
    q_table_df.index = [
        environment.get_position(state) for state in range(q_table_df.shape[0])
    ]
    q_table_df.to_csv("output/mc_q_table.csv")
    if plot_simulation:
        utils.plot_simulation_results(sim_input, sim_output)
    return sim_output


if __name__ == "__main__":
    arg_parser = utils.get_argument_parser()
    arg_parser.add_argument("--first_visit", action="store_true")
    args = arg_parser.parse_args()
    main(
        args.num_episodes,
        args.gamma,
        args.alpha,
        args.epsilon,
        args.first_visit,
        plot_simulation=True,
    )
