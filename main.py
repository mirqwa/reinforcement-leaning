from learning_algorithms import sarsa, q_learning

num_of_episodes = 1000
gamma = 1
alpha = 0.1
epsilon = 0.1

sarsa.main(num_of_episodes, gamma, alpha, epsilon)
q_learning.main(num_of_episodes, gamma, alpha, epsilon)
