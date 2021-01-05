import gym
import argparse
from gym import wrappers, logger


import torch
import random
import skimage
import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple

import math


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
        self.r_sum = 0

    def add_value(self, reward):
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
# a generale buffer from pytorch doc.
    
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
        self.fc_hidden = torch.nn.Linear(h_l_size, h_l_size)
        self.fc_out = torch.nn.Linear(h_l_size, output_size)

        self.relu = torch.nn.functional.relu
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2)


    def forward(self, x):
        x = self.leaky_relu(self.fc_in(x))
        x = self.relu(self.fc_hidden(x))
        return self.fc_out(x)


class Agent():

    def __init__(self, env: gym.Env, alpha, update_count, epsilon_proba, lr):
        self.ob_space = env.observation_space # espace des états
        self.a_space = env.action_space # espace des actions
        self.epsilon_proba = epsilon_proba # for greedy exploration
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.08
        self.alpha = alpha
        self.count_learn = 0
        self.update_count = update_count
        self.train_mod = True


    def preprocess(self, ob):
        return ob


    def load(self, path):
        print('model loaded from:', path)
        self.train_mod = False

        checkpoint = torch.load(path)
        # self.epsilon_proba = checkpoint['epsilon_proba']
        print('epoch', checkpoint['epoch'])

        self.net.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.net.eval()


    def save(self, path, epoch):
        checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.net.state_dict(),
                # 'optimizer_state_dict': self.optimizer.state_dict(),
                # 'epsilon_proba': self.epsilon_proba,
            }
        torch.save(checkpoint, path)

    # 2.3.6
    def greedy_exploration(self, qvalues, debug) -> int:

        if self.train_mod and np.random.rand() < self.epsilon_proba:
            return self.a_space.sample()

        action = torch.argmax(qvalues)
        if debug:  print(action)
        return int(action)


    #2.3.6
    def act(self, ob, reward, done, debug=False):

        # au début on explore bcp avec epsilon_proba=1
        # puis on limite exploration et on se concentre plus sur les meilleurs actions

        inputs = torch.tensor([ob]).float()
        
        if self.train_mod:
            self.net.eval() # eval mod
            with torch.no_grad():  qvaleurs = self.net(inputs)
            self.net.train() # train mod
        else:
            with torch.no_grad():  qvaleurs = self.net(inputs)

        if debug:
            print(inputs)
            print(qvaleurs)
            input('next?')

        return self.greedy_exploration(qvaleurs, debug)


    def train(self, buffer: Buffer, sample_size: int, gamma: float, debug=False):
        
        if self.train_mod and self.epsilon_proba > self.epsilon_min:
            self.epsilon_proba *= self.epsilon_decay

        samples = buffer.get_sample(sample_size)

        inputs = torch.tensor([ s.state for s in samples ]).float()
        next_states = torch.tensor([ s.next_state for s in samples ]).float()
        dones = torch.tensor([ 0 if s.done else 1 for s in samples ], dtype=torch.int8).unsqueeze(1)
        rewards = torch.tensor([ s.reward for s in samples ]).unsqueeze(1)
        actions_took = torch.tensor([ s.action for s in samples ]).unsqueeze(1)

        # le model calcule q(s) puis on selectionne q(s, a) selon les action prise
        qvalues = self.net(inputs).gather(1, actions_took)
        
        # print(qvalues.shape)

        self.target_net.eval()
        with torch.no_grad(): 
            next_states_qvs = self.target_net(next_states).max(1).values.unsqueeze(1)

        labels = rewards + (gamma * next_states_qvs * dones)

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


class CartPoleAgent(Agent):

    def __init__(self, env: gym.Env, alpha=0.01, update_count=500, epsilon_proba=1.0, lr=0.005):

        super().__init__(env, alpha, update_count, epsilon_proba, lr)

        self.net = FCModel(self.ob_space.shape[0], self.a_space.n)
        self.target_net = FCModel(self.ob_space.shape[0], self.a_space.n)
        self.target_net.load_state_dict(self.net.state_dict())

        self.loss_func = torch.nn.MSELoss() # the mean squared error
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)


class CnnModel(torch.nn.Module):

    def __init__(self, h, w, output_size):
        super(CnnModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = torch.nn.Conv2d(6, 6, kernel_size=5, stride=1)
        # self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=5, stride=1)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 6

        self.head = torch.nn.Linear(linear_input_size, output_size)

        self.relu = torch.nn.functional.relu

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        # x = self.relu(self.conv3(x))
        return self.head(x.view(x.size(0), -1))


class VizDoomAgent(Agent):

    def __init__(self, env: gym.Env, alpha=0.005, u_c=500, eps=0.1, lr=0.005, res=(112, 64, 3)): # (112, 64, 3) (28, 16, 3)
        super().__init__(env, alpha, u_c, eps, lr)
        self.resolution = res

        self.net = CnnModel(res[0], res[1], self.a_space.n)
        self.target_net = CnnModel(res[0], res[1], self.a_space.n)
        self.target_net.load_state_dict(self.net.state_dict())

        self.loss_func = torch.nn.MSELoss() # the mean squared error
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)


    def preprocess(self, img):
        img = img[0]
        img = skimage.transform.resize(img, self.resolution)
        #passage en noir et blanc
        img = skimage.color.rgb2gray(img)
        #passage en format utilisable par pytorch
        img = img.astype(np.float32)
        # print(img.shape)
        img = img.reshape((1, self.resolution[0], self.resolution[1]))
        return img



def evaluate_agent(env, agent, episode_count, render=False):

    reward = 0
    done = False
    reword_recorder = RewordRecorder()

    actions = {}


    for i in range(episode_count):
        ob = env.reset()
        step = 0
        reword_recorder.start_episode()
        while True:

            action = agent.act(agent.preprocess(ob), reward, done)

            if action in actions.keys():
                actions[action] += 1
            else:
                actions[action] = 0

            ob, reward, done, _ = env.step(action)

            reword_recorder.add_value(reward)
            if render:  env.render()

            step += 1
            if done:
                if i % 100 == 0:
                    print('episode {} done took {} steps'.format(i, step))
                reword_recorder.recorde_episode()
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    print('mean:', np.mean(reword_recorder.rewards))
    print('max:', np.max(reword_recorder.rewards))
    print('min:', np.min(reword_recorder.rewards))

    return reword_recorder, actions



def train_agent(env, agent, config):

    gamma = config['gamma']
    best_path = config['best_path']
    save_model = config['save_model']
    batch_size = config['batch_size']
    skip_frame = config['skip_frame']
    buffer_size = config['buffer_size']
    episode_count = config['episode_count']
    

    reward = 0
    done = False
    best_reward = -math.inf
    reword_recorder = RewordRecorder()

    buffer = Buffer(buffer_size)
    actions = {}


    for i in range(episode_count):
        ob = env.reset()
        reword_recorder.start_episode()
        step = 0
        while True:

            action = agent.act(agent.preprocess(ob), reward, done)

            if action in actions.keys():  actions[action] += 1
            else:  actions[action] = 0


            if step % skip_frame == 0:
                interaction = Interaction(agent.preprocess(ob), action, None, None, None)
                ob, reward, done, _ = env.step(action)
                interaction = Interaction(interaction.state, interaction.action, agent.preprocess(ob), reward, done)

                buffer.add_value(interaction)

                if len(buffer) > batch_size:  
                    agent.train(buffer, batch_size, gamma)
            else:
                _, reward, done, _ = env.step(action)

            reword_recorder.add_value(reward)
            step += 1

            if done:
                reword_recorder.recorde_episode()
                if i % 50 == 0:
                    print('episode {} done took {} steps and reward {} '.format(i, step, reword_recorder.rewards[-1]))
                
                if save_model and best_reward <= reword_recorder.rewards[-1]:
                    best_reward = reword_recorder.rewards[-1]
                    agent.save(best_path, i)

                break
    


    return reword_recorder, actions


if __name__ == '__main__':

    _buffer_test()
