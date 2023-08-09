import argparse

import plot


class sim_init:
    def __init__(self, num_episodes, gamma, alpha, epsilon):
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def __str__(self):
        return (
            "# episodes: "
            + str(self.num_episodes)
            + "gamma: "
            + str(self.gamma)
            + "alpha: "
            + str(self.alpha)
            + "epsilon: "
            + str(self.epsilon)
        )


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
    plot.plot_data(sim_output.step_cache, sim_output.name_cache, "# steps")
    plot.plot_data(sim_output.reward_cache, sim_output.name_cache, "rewards")
    plot.plot_path(sim_output)


def get_argument_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--num_episodes", default=1000, type=int)
    arg_parser.add_argument("--gamma", default=1, type=float)
    arg_parser.add_argument("--alpha", default=0.1, type=float)
    arg_parser.add_argument("--epsilon", default=0.1, type=float)
    return arg_parser
