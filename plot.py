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


def plot_data(data_cache, data_cache_names, y_label):
    sns.set_theme(style="darkgrid")
    positions, labels, batch_to_be_averaged = get_plot_positions_and_labels(
        data_cache[0]
    )
    for i in range(len(data_cache)):
        mod = len(data_cache[i]) % batch_to_be_averaged
        mean_step = np.mean(
            data_cache[i][mod:].reshape(-1, batch_to_be_averaged), axis=1
        )
        sns.lineplot(data=mean_step, label=data_cache_names[i])

    # Plot graph
    plt.xticks(positions, labels)
    plt.ylabel(y_label)
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
