import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from neural_network import NeuralNetwork
from constants import learning_rate, replay_buffer_size, minibatch_size, discount_factor, interpolation_parameter
from replay_memory import ReplayMemory


class Agent(object):
    def __init__(self, state_size: int, action_size: int):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        # Variables to implement Q-Learning
        # local_q_nnetwork is the one that selects the actions
        self.local_q_nnetwork = NeuralNetwork(state_size, action_size).to(self.device)
        # target_q_nnetwork is the one that calculates the Q values that will train the local_q_nnetwork
        self.target_q_nnetwork = NeuralNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_q_nnetwork.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(replay_buffer_size)
        self.time_step = 0

    def step(self, state, action, reward, next_state, done):
        """
        Handles when to learn from an experience
        *All params put together create an "experience"
        """
        self.memory.push((state, action, reward, next_state, done))
        # We want to reset the step counter every 4 steps and to do that we check
        # if the division between counter + 1 and 4 = 0
        self.time_step = (self.time_step + 1) % 4
        if self.time_step == 0:
            if len(self.memory.memory) > minibatch_size:
                experiences = self.memory.sample(minibatch_size)
                self.learn(experiences, discount_factor)

    def act(self, state, epsilon=0.):
        """
        Handles selecting an action based on a given state following the epsilon greedy action
        selection policy. This policy is based on exploration strategies in order to allow the agent
        to explore several different actions

        This policy works as follows:
        We generate a random number and if that number is greater than epsilon, we will return
        the action with the highest Q value from the action_values gathered from our local_q_nnetwork
        BUT if the random number is lower than epsilon, we will return a random action from our
        action pool (action_size)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_q_nnetwork.eval()
        with torch.no_grad():
            action_values = self.local_q_nnetwork(state)
        self.local_q_nnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        Handles how to teach our agent to perform actions
        """
        states, next_states, actions, rewards, dones = experiences
        # We propagate the next state to the target Q nn
        next_q_targets = self.target_q_nnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        # Implementation of the XXXX math formula
        q_targets = rewards + (gamma * next_q_targets * (1 - dones))
        q_expected = self.local_q_nnetwork(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        # Updates model parameters
        self.optimizer.step()
        self.soft_update(self.local_q_nnetwork, self.target_q_nnetwork, interpolation_parameter)

    def soft_update(self, local_q_nnetwork, target_q_nnetwork, _interpolation_parameter):
        """
        Handles softly updating the target network parameters
        """
        for target_params, local_params in zip(target_q_nnetwork.parameters(), local_q_nnetwork.parameters()):
            target_params.data.copy_(_interpolation_parameter * local_params.data + (1.0 - _interpolation_parameter) *
                                     target_params.data)
