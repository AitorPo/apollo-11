from collections import deque

import gymnasium as gym

# Lunar Lander environment variables from Gymnasium documentation
env = gym.make("LunarLander-v2")
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
learning_rate = 5e-4  # 5·10^-4 = 0.0005. Known with trial error
minibatch_size = 100  # Common value for Deep Q-Learning
"""
discount_factor represents the present value of future rewards.
If we have a value close to 0 the agent will only consider the current rewards without
considering the future rewards. So, the closer to 1, the better te results because the
AKA gamma
"""
discount_factor = 0.99
replay_buffer_size = int(1e5)  # 10^5 = 100000. Represents the memory of the IA
interpolation_parameter = 1e-3  # 10^-3 = 0.001. AKA TAO. ¡¡¡¡DO NOT CAST TO INT!!!!

# Variables used in the training
number_episodes = 2000
max_steps_per_episode = 1000
epsilon_start_value = 1.0
epsilon_end_value = 0.01
epsilon_decay_value = 0.995
epsilon = epsilon_start_value
scores = deque(maxlen=100)
