from environment import RandomWalk


# experiment parameters
N_EPISODES = 100000
ALPHA = 2e-5
N_STATE_BINS = 10  # bins for state aggregation
NUM_STATES = 1000
STATE_BIN_SIZE = NUM_STATES / N_STATE_BINS


def gradient_mc():
    mdp = RandomWalk()


if __name__ == "__main__":
    gradient_mc()
