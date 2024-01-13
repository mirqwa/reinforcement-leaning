import plot
from learning_algorithms import sarsa, q_learning, expected_sarsa


num_of_episodes = 5
gamma = 1
alpha = 0.1
epsilon = 0.1

sarsa_sim_output = sarsa.main(num_of_episodes, gamma, alpha, epsilon)
q_learning_sim_output = q_learning.main(num_of_episodes, gamma, alpha, epsilon)
expected_sarsa_sim_output = expected_sarsa.main(num_of_episodes, gamma, alpha, epsilon)
steps_cache = [
    sarsa_sim_output.step_cache[0],
    q_learning_sim_output.step_cache[0],
    expected_sarsa_sim_output.step_cache[0],
]
names_cache = [
    sarsa_sim_output.name_cache[0],
    q_learning_sim_output.name_cache[0],
    expected_sarsa_sim_output.name_cache[0],
]
rewards_cache = [
    sarsa_sim_output.reward_cache[0],
    q_learning_sim_output.reward_cache[0],
    expected_sarsa_sim_output.reward_cache[0],
]
plot.plot_data(steps_cache, names_cache, "# steps")
plot.plot_data(rewards_cache, names_cache, "rewards")
