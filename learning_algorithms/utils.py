import plot


class sim_init:
    def __init__(self, num_episodes, gamma, alpha, epsilon):
        self.num_episodes = num_episodes  # Number of training episodes
        self.gamma = gamma  # Discount rate γ 0.9
        self.alpha = alpha  # Learning rate α 0.001
        self.epsilon = epsilon  # Exploration rate ε

    def __str__(self):
        return "# episodes: " + str(self.num_episodes) + "gamma: " + str(self.gamma) \
                + "alpha: " + str(self.alpha) + "epsilon: " + str(self.epsilon)


class sim_output:
    def __init__(self, rewards_cache, step_cache, env_cache, name_cache):
        self.reward_cache = rewards_cache  # list of rewards
        self.step_cache = step_cache  # list of steps
        self.env_cache = env_cache  # list of final paths
        self.name_cache = name_cache  # list of algorithm names


def plot_simulation_results(sim_input, sim_output):
    plot.console_output(
        sim_output,
        sim_input.num_episodes,
    )
    # Plot output
    plot.plot_steps(sim_output)
    plot.plot_rewards(sim_output)
    plot.plot_path(sim_output)
