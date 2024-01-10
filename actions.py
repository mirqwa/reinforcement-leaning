import numpy as np


def epsilon_greedy_action(state: int, q_table: np.array, epsilon: float) -> int:
    """
    Select action based on the ε-greedy policy
    Random action with prob. ε, greedy action with prob. 1-ε
    """
    # explore of random value <= ε
    if np.random.random() <= epsilon:  # Explore
        return np.random.choice(4)
    return np.argmax(q_table[:, state])


def move_agent(agent_pos: tuple, action: int) -> tuple:
    """
    Move agent to new position based on current position and action
    """
    # Retrieve agent position
    (pos_x, pos_y) = agent_pos
    if action == 0:  # Up
        pos_x = pos_x - 1 if pos_x > 0 else pos_x
    elif action == 1:  # Down
        pos_x = pos_x + 1 if pos_x < 3 else pos_x
    elif action == 2:  # Left
        pos_y = pos_y - 1 if pos_y > 0 else pos_y
    elif action == 3:  # Right
        pos_y = pos_y + 1 if pos_y < 11 else pos_y
    else:  # Infeasible move
        raise Exception("Infeasible move")
    return pos_x, pos_y


def get_max_qvalue(state: int, q_table: np.array) -> float:
    """Retrieve best Q-value for state from table"""
    maximum_state_value = np.amax(q_table[:, state])
    return maximum_state_value


def get_reward(state: int, cliff_pos: np.array, goal_pos: int) -> int:
    """
    Compute reward for given state
    """
    return -100 if state in cliff_pos else -1

def compute_cum_rewards(gamma: float, t: int, rewards: np.array) -> float:
    """Cumulative reward function"""
    cum_reward = 0
    for tau in range(t, len(rewards)):
        cum_reward += gamma ** (tau - t) * rewards[tau]
    return cum_reward
