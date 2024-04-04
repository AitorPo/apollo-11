import random

import numpy as np
import torch


class ReplayMemory(object):
    def __init__(self, capacity: int):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity  # Maximum size of the memory buffer
        self.memory = []  # List that will store the experiences/events that our agent is "living"

    def push(self, event):
        """Handles adding an event to the memory list taking care of buffer's memory size.
        If the memory exceeds the buffer's size we will delete the oldest event from the memory because that's the less
        relevant
        """
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size: int):
        """

        :param batch_size: number of experiences/events that are going to be sampled in the batch
        :return:
        """
        experiences = random.sample(self.memory, k=batch_size)
        # We gather all experiences in the batch and send them to the computational device
        # experiences are in the first index
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        # actions are in the second position. Their type is going to be a long value (int) as long as actions are designed as 0, 1, 2...
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        # rewards are in the third position
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        # upcoming states such as velocity, angles, coordinates...
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        # np.uint8 represents boolean values before converting them to the designated type: float in this case
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return states, next_states, actions, rewards, dones
