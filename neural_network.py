import torch

import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    """
    Class that represents the "brain" of our AI
    """

    def __init__(self, state_size: int, action_size: int, seed: int = 42) -> None:
        """
        fc1 => This layer is directly connected to the input layer. Its shape is composed by the state_size and a proper
               number of neurons to perform the actions as we want
        fc2 => Hidden layer which shape is composed by the same number of output neurons of the previous layer and the
               number of neurons to perform the actions as we want. If we do not need more layers we would have closed
               the shape using action_size
        fc3 => Output layer which shape is composed by the number of output neurons of the previous layer and the actual
               action_size in order to "close" the NN.

        :param state_size:
        :param action_size: possible actions.
        :param seed:
        """
        super(NeuralNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)  # generates random numbers in the range of "seed"
        self.fc1 = nn.Linear(state_size, 64*3)  # first fully connected layer. Directly connected to I. Layer
        self.fc2 = nn.Linear(64*3, 64*3)  # second fully connected layer
        self.fc3 = nn.Linear(64*3, action_size)  # third fully connected layer: Output layer

    def forward(self, state):
        """
        Handles propagating the signal from the input layer through fully connected layers until the output layer that
        contains the action_state
        :param state:
        :return:
        """
        # Passes the state from the input layer to the first fully connected layer
        x = self.fc1(state)
        x = F.relu(x)
        # Passes the state from the first fully connected layer to the second fully connected layer
        x = self.fc2(x)
        x = F.relu(x)
        # Returns the signal to the output layer
        return self.fc3(x)
