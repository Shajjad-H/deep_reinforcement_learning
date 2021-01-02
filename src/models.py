import gym
import argparse
from gym import wrappers, logger


import torch
import numpy as np
import random
import matplotlib.pyplot as plt

from collections import namedtuple


# 2.1.1
class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


# 2.1.2
class RewordRecorder():

    def __init__(self) -> None:
        self.rewards = []

    def start_episode(self):
        self.c = 0
        self.r_sum = 0

    def add_value(self, reward):
        self.c += 1
        self.r_sum += reward

    def recorde_episode(self):
        self.rewards.append(self.r_sum)

    def show(self):
        plt.plot(self.rewards)
        plt.ylabel('rewards par episode')
        plt.xlabel('episodes')
        plt.show()


# 2.2.3
class Buffer():
# a generale buffer.
    
    def __init__(self, buffer_size) -> None:
        self.buffer_size = buffer_size
        self.buffer = []
        self.ci = 0 # current index

    # 2.2.4
    def get_sample(self, size) -> list:
        if len(self.buffer) < size: # the sample size can't be bigger than the buffer
            print('Warning: simple size bigger than buffer size')
            size = len(self.buffer)

        return random.sample(self.buffer, size)


    def __len__(self) -> int:
        return len(self.buffer)

    def add_value(self, value) -> None:
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(value)
        else:
            self.buffer[self.ci] = value

        self.ci = (self.ci + 1) % self.buffer_size



def _buffer_test():
    buffer = Buffer(10)
    
    for i in range(100):
        buffer.add_value((i, i ** 2, i % 2 == 0))

    last_part = []
    for i in range(90, 100, 1):
        last_part.append((i, i ** 2, i % 2 == 0))
    
    last_part = np.array(last_part)

    assert np.all(buffer.buffer == last_part) == True

Interaction = namedtuple(typename='Interaction', field_names=['state', 'action', 'next_state', 'reward', 'done'])



"""
    fully conected layer to approximate qvalue.
    q 2.3.5
"""
class FCModel(torch.nn.Module):

    def __init__(self, input_size, output_size, hidden_layer_size=None) -> None:
        super(FCModel, self).__init__()
        h_l_size = input_size * output_size if hidden_layer_size is None else hidden_layer_size

        print('initializing FCModel input size:', input_size, 'output size:', output_size, 'hidden layer:', h_l_size)

        self.fc_in = torch.nn.Linear(input_size, h_l_size)
        # self.fc_hidden = torch.nn.Linear(h_l_size, h_l_size)
        self.fc_out = torch.nn.Linear(h_l_size, output_size)

        self.relu = torch.nn.functional.relu
        # self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2)


    def forward(self, x):
        x = self.relu(self.fc_in(x))
        # x = self.relu(self.fc_hidden(x))
        return self.fc_out(x)


class CartPoleAgent():

    def __init__(self, env: gym.Env, alpha=0.005, update_count=500, epsilon_proba=0.1, lr=0.005) -> None:
        
        self.ob_space = env.observation_space # espace des états
        self.a_space = env.action_space # espace des actions
        self.epsilon_proba = epsilon_proba # for greedy exploration
        self.alpha = alpha

        self.net = FCModel(self.ob_space.shape[0], self.a_space.n)
        self.target_net = FCModel(self.ob_space.shape[0], self.a_space.n)
        self.target_net.load_state_dict(self.net.state_dict())

        self.count_learn = 0
        self.update_count = update_count

        self.loss_func = torch.nn.MSELoss() # the mean squared error
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)


    def train(self, buffer: Buffer, sample_size: int, gamma: float, debug=False):

        samples = buffer.get_sample(sample_size)

        inputs = torch.tensor([ s.state for s in samples ]).float()
        next_states = torch.tensor([ s.next_state for s in samples ]).float()
        dones = torch.tensor([ 0 if s.done else 1 for s in samples ], dtype=torch.int8).unsqueeze(1)
        rewards = torch.tensor([ s.reward for s in samples ]).unsqueeze(1)
        actions_took = torch.tensor([ s.action for s in samples ]).unsqueeze(1)

        # le model calcule q(s) puis on selectionne q(s, a) selon les action prise
        qvalues = self.net(inputs).gather(1, actions_took)

        self.target_net.eval()
        with torch.no_grad(): 
            next_states_qvs = self.target_net(next_states).max(1).values.unsqueeze(1)

        labels = rewards #+ (gamma * next_states_qvs * dones)

        if debug:
            print('samples', samples[0])
            print('inputs', inputs[:5])
            print('qvalues', qvalues[:5])
            print('rewards', rewards[:5])
            print('next_states_max_qvs', next_states_qvs[:5])
            print('labels', labels[:5])
            input('next?')
        
        # zero the parameter gradients
        self.optimizer.zero_grad()
        # forward + backward + optimize
        loss = self.loss_func(qvalues, labels) # (Qθ(s, a) − (r(s, a) + γ maxa0 Qˆθ(s0, a0))^2
        loss.backward()


        self.optimizer.step()

        # self.count_learn += 1
        # if self.update_count < self.count_learn:
        #     self.count_learn = 0
        self.target_net.load_state_dict(self.copy_model())#self.net.state_dict())


    def copy_model(self):
        eval_dict = self.net.state_dict()
        target_dict = self.net.state_dict()

        for weights in eval_dict:
            target_dict[weights] = (1 - self.alpha) * target_dict[weights] + self.alpha * eval_dict[weights]

        return target_dict

    # 2.3.6
    def greedy_exploration(self, qvalues, debug):
        
        if np.random.rand() < self.epsilon_proba: return self.a_space.sample()

        action = torch.argmax(qvalues)

        if debug:  print(action)

        return int(action)

    #2.3.6
    def act(self, ob, reward, done, debug=False):

        inputs = torch.tensor([ob]).float()

        self.net.eval() # eval mod
        with torch.no_grad():  qvaleurs = self.net(inputs)
        self.net.train() # train mod

        if debug:
            print(inputs)
            print(qvaleurs)
            input('next?')

        return self.greedy_exploration(qvaleurs, debug)



if __name__ == '__main__':

    _buffer_test()

 