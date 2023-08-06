import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from environment import env_to_text


def get_plot_positions_and_labels(steps_cache):
    step_size = int(len(steps_cache) / 10)
    positions = np.arange(0, 100, 10)
    labels = np.arange(0, len(steps_cache), step_size)
    batch_to_be_averaged = int(len(steps_cache) / 100)
    return positions, labels, batch_to_be_averaged


def plot_rewards(sim_output) -> None:
    """
    Visualizes rewards
    """
    sns.set_theme(style="darkgrid")
    # Set x-axis label
    positions, labels, batch_to_be_averaged = get_plot_positions_and_labels(
        sim_output.step_cache[0]
    )

    for i in range(len(sim_output.step_cache)):
        mod = len(sim_output.reward_cache[i]) % batch_to_be_averaged
        mean_reward = np.mean(
            sim_output.reward_cache[i][mod:].reshape(-1, batch_to_be_averaged), axis=1
        )
        sns.lineplot(data=mean_reward, label=sim_output.name_cache[i])

    # Plot graph
    plt.xticks(positions, labels)
    plt.ylabel("rewards")
    plt.xlabel("# episodes")
    plt.legend(loc="best")

    plt.show()

    return


def plot_steps(
    sim_output,
) -> None:
    """
    Visualize number of steps taken
    """
    sns.set_theme(style="darkgrid")
    positions, labels, batch_to_be_averaged = get_plot_positions_and_labels(
        sim_output.step_cache[0]
    )

    for i in range(len(sim_output.step_cache)):
        mod = len(sim_output.step_cache[i]) % batch_to_be_averaged
        mean_step = np.mean(
            sim_output.step_cache[i][mod:].reshape(-1, batch_to_be_averaged), axis=1
        )
        sns.lineplot(data=mean_step, label=sim_output.name_cache[i])

    # Plot graph
    plt.xticks(positions, labels)
    plt.ylabel("# steps")
    plt.xlabel("# episodes")
    plt.legend(loc="best")
    plt.show()


def console_output(
    sim_output,
    num_episodes: int,
) -> None:
    """Print path and key metrics in console"""
    for i in range(len(sim_output.env_cache)):
        env_str = env_to_text(sim_output.env_cache[i])

        print("=====", sim_output.name_cache[i], "=====")
        print("Action after {} iterations:".format(num_episodes), "\n")
        print(env_str, "\n")
        print(
            "Number of steps:", int(sim_output.step_cache[i][-1]), "(best = 13)", "\n"
        )
        print("Reward:", int(sim_output.reward_cache[i][-1]), "(best = -2)", "\n")


def plot_path(
    sim_output,
) -> None:
    """Plot latest paths as heatmap"""

    # Set values for cliff
    for i in range(len(sim_output.env_cache)):
        for j in range(1, 11):
            sim_output.env_cache[i][3, j] = -1

        ax = sns.heatmap(
            sim_output.env_cache[i],
            square=True,
            cbar=True,
            xticklabels=False,
            yticklabels=False,
        )
        ax.set_title(sim_output.name_cache[i])
        plt.show()
