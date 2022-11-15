#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, in_channels=4, num_actions=4):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #

        """
        This network is an exact replication of the DQN network used in human-level control.
        """

        # CONV layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1))

        # LINEAR layers
        self.linear1 = nn.Linear(in_features=7*7*64, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=num_actions)


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 7*7*64)

        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        ###########################
        return x
