#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys
import random

import torch
import torch.nn.functional as F
import torch.optim as optim 

from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

_seed = 525
torch.manual_seed(_seed)
np.random.seed(_seed)
random.seed(_seed)

'''
HYPERPARAMETERS
'''

# RL hyperparameters
MINIBATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 200000
MIN_REPLAY_MEMORY = 25000
TARGET_UPDATE_FREQUENCY = 5000
DISCOUNT_FACTOR = 0.99

# action repetition hyperparameters
ACTION_REPEAT = 4
UPDATE_FREQUENCY = 4

# optimizer hyperparameters
LEARNING_RATE = 2.5e-4
GRADIENT_MOMENTUM = 0.95
MIN_SQUARED_GRADIENT = 0.01

# epsilon greedy hyperparameters
EPSILON_START = 1.0
EPSILON_END = 0.1
LAST_EPSILON_STEP = 750000

# model training hyperparameters
TRAIN_EPISODES = 50000
SAVE_FREQUENCY = 1000

'''
END HYPERPARAMETERS
'''

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #

        if args.use_cuda:
            assert torch.cuda.is_available(), "Applied --use_cuda but no CUDA device available."

            self.use_cuda = True
            self.cuda_device = torch.device('cuda')
        else:
            self.use_cuda = False

        self.cpu_device = torch.device('cpu')

        # Initialize policy_dqn and target_dqn to be the same network
        self.policy_dqn = DQN()
        self.target_dqn = DQN()

        torch.save(self.policy_dqn, "policy_dqn.pt")

        if self.use_cuda:
            self.policy_dqn.to(self.cuda_device)
            self.target_dqn.to(self.cuda_device)

        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        self.target_dqn.eval()

        # start with epsilon
        self.epsilon = EPSILON_START

        # initialize replay buffer
        self.replay = deque(maxlen = REPLAY_MEMORY_SIZE)

        # initialize optimizer
        self.optimizer = optim.Adam(self.policy_dqn.parameters(), lr=LEARNING_RATE)

        # initialize loss function
        self.loss_fn = torch.nn.SmoothL1Loss()

        # current epsilon step: we want to stop decaying after reaching LAST_EPSILON_STEP
        self.epsilon_step = 0

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            self.policy_dqn = torch.load("policy_dqn.pt", map_location=self.cpu_device)
            self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        # compute y_pred either way to know the action shape
        with torch.no_grad():
            if test:
                y_pred = self.policy_dqn.forward(torch.Tensor(observation))
            else:
                y_pred = self.policy_dqn.forward(observation)

            if (random.random() < self.epsilon) and not test:
                action = np.random.randint(0, y_pred.shape[1])
            else:
                if self.use_cuda:
                    action = np.argmax(y_pred.detach().cpu().numpy()[0])
                else:
                    action = np.argmax(y_pred.detach().numpy()[0])

            if not test:
                if self.epsilon_step < LAST_EPSILON_STEP:
                    self.epsilon -= (EPSILON_START - EPSILON_END) / LAST_EPSILON_STEP # linear decay
                    self.epsilon_step += 1

        ###########################
        return action
    
    def push(self, transition):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.replay.append(transition) # lol
        ###########################
        
        
    def replay_buffer(self, batch_size):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        return random.sample(self.replay, batch_size)
        

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        reward_array = []

        for _ep in range(TRAIN_EPISODES):

            # begin Q-learning episode
            observation = self.env.reset() # observation will always be a frame of shape (84, 84, 4)
            observation = torch.permute(torch.Tensor(observation), (2, 0, 1)) / 255.0

            done = False
            total_reward = 0
            episode_length = 0

            while not done:
                episode_length += 1

                observation_old = observation # s

                if self.use_cuda:
                    action = self.make_action(observation_old.to(self.cuda_device), test=False) # a with cuda
                else:
                    action = self.make_action(observation_old, test=False) # a without cuda

                observation, reward, done, _, _ = self.env.step(action) # r, s'
            
                observation = torch.permute(torch.Tensor(observation), (2, 0, 1)) / 255.0
                total_reward += reward # accumulate reward

                self.push((observation_old, action, reward, observation)) # push (s, a, r, s') to buffer


                if len(self.replay) < MIN_REPLAY_MEMORY: # need enough data in buffer to get a batch
                    continue

                batch_transitions = self.replay_buffer(MINIBATCH_SIZE) # get a batch from buffer

                batch_states = torch.stack([transition[0] for transition in batch_transitions]) # get s from each transition as a Tensor
                batch_actions = torch.Tensor([transition[1] for transition in batch_transitions]).type(torch.int64) # get a from each transition as a Tensor
                batch_rewards = torch.Tensor([transition[2] for transition in batch_transitions]) # get all rewards as a Tensor
                batch_next_states = torch.stack([transition[3] for transition in batch_transitions]) # get s' from each transition

                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_states)), dtype=torch.bool) 
                non_final_next_states = torch.stack([s for s in batch_next_states if s is not None])

                max_next_values = torch.zeros(MINIBATCH_SIZE)

                if self.use_cuda:
                    batch_states = batch_states.to(self.cuda_device)
                    batch_actions = batch_actions.to(self.cuda_device)
                    batch_rewards = batch_rewards.to(self.cuda_device)
                    batch_next_states = batch_next_states.to(self.cuda_device)
                    non_final_mask = non_final_mask.to(self.cuda_device)
                    non_final_next_states = non_final_next_states.to(self.cuda_device)
                    max_next_values = max_next_values.to(self.cuda_device)

                max_next_values[non_final_mask] = self.target_dqn.forward(non_final_next_states).max(1)[0].detach() # apply DQN net to next_states and get max Q

                y = batch_rewards + DISCOUNT_FACTOR * max_next_values # ground truth values

                y_pred = self.policy_dqn.forward(batch_states).gather(1, batch_actions.view(-1, 1)) # predictions of value function

                loss = self.loss_fn(y_pred, y.unsqueeze(1))

                self.optimizer.zero_grad()
                loss.backward()
                for param in self.policy_dqn.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

            reward_array.append(total_reward)
            print("Reward:", total_reward, "[" + str(_ep) + "/" + str(TRAIN_EPISODES) + ", episode length " + str(episode_length) + "], epsilon = " + str(round(self.epsilon, 2)))

            if _ep % SAVE_FREQUENCY == 0:
                torch.save(self.policy_dqn, "policy_dqn.pt")

                # output reward averaging
                next_reward_array = np.array(reward_array)
                with open("rewards.txt", "w") as f:
                    for j in range(30, len(next_reward_array)):
                        f.write(str(np.mean(next_reward_array[j - 30:j])) + "\n")

            if _ep % TARGET_UPDATE_FREQUENCY == 0:
                self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

        # save model
        torch.save(self.policy_dqn, "policy_dqn.pt")
        ###########################
