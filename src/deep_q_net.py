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
    def get_sample(self, size):
        if len(self.buffer) < size: # the sample size can't be bigger than the buffer
            print('Warning: simple size bigger than buffer size')
            size = len(self.buffer)

        return random.sample(self.buffer, size)


    def __len__(self) -> int:
        return len(self.buffer)

    def add_value(self, value):
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
        h_l_size = ((input_size * output_size)) if hidden_layer_size is None else hidden_layer_size

        self.fc_in = torch.nn.Linear(input_size, h_l_size)
        # self.fc_hidden = torch.nn.Linear(h_l_size, h_l_size)
        self.fc_out = torch.nn.Linear(h_l_size, output_size)

        self.relu = torch.nn.functional.relu
        self.leaky_relu = torch.nn.functional.leaky_relu
        self.sigmaoide = torch.sigmoid

    def forward(self, x):
        # x = self.leaky_relu(self.fc_in(x))
        x = self.relu(self.fc_in(x))
        # x = self.relu(self.fc_hidden(x))
        return self.sigmaoide(self.fc_out(x))


class DeepNetAgent():

    def __init__(self, env: gym.Env, nn_model, epsilon_proba=0.2, lr:float=0.001, momentum:float=0.9) -> None:
        
        self.ob_space = env.observation_space # espace des états
        self.a_space = env.action_space # espace des actions
        self.epsilon_proba = epsilon_proba # for greedy exploration

        self.net = nn_model(self.ob_space.shape[0], self.a_space.n)
        self.criterion = torch.nn.L1Loss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=momentum)


    def train(self, buffer: Buffer, sample_size: int, alpha: float, debug=False):

        samples = buffer.get_sample(sample_size)

        inputs = torch.tensor([ s.state for s in samples ]).float()
        next_states = torch.tensor([ s.next_state for s in samples ]).float()
        dones = torch.tensor([ 0 if s.done else 1 for s in samples ]).reshape((len(samples),1))
        rewards = torch.tensor([ s.reward for s in samples ]).reshape((len(samples),1))

        qvalues = self.net(inputs)
        next_states_qvs = self.net(next_states)
        next_states_max_qvs = torch.max(next_states_qvs, dim=1).values.reshape((len(samples),1))

        # print(qvalues.shape)
        # print(rewards.shape)
        # print(dones.shape)
        # print('next_states_max_qvs', next_states_max_qvs.shape)


        labels = qvalues - (rewards + alpha * next_states_max_qvs * dones)

        if debug:
            print('samples', samples)
            print('inputs', inputs)
            print('qvalues', qvalues)
            print('labels', labels)
            input('next?')

        # zero the parameter gradients

        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
    
    # 2.3.6
    def greedy_exploration(self, qvalues, debug):
        if np.random.rand() < self.epsilon_proba:
            random_action = np.random.choice(self.a_space.n, size=1, replace=False)
            return random_action[0]

        action = torch.argmax(qvalues)
        
        if debug:
            print(action)

        return int(action)

    #2.3.6
    def act(self, ob, reward, done, debug=False):
        inputs = torch.tensor([ob]).float()
        qvaleurs = self.net(inputs)

        if debug:
            print(inputs)
            print(qvaleurs)
            input('next?')

        return self.greedy_exploration(qvaleurs, debug)


if __name__ == '__main__':

    _buffer_test()

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    # env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = DeepNetAgent(env, FCModel)

    episode_count = 1000
    reward = 0
    done = False
    reword_recorder = RewordRecorder()

    BATCH_SIZE = 128
    ALPHA = 0.6

    buffer = Buffer(10000)

    for i in range(episode_count):
        ob = env.reset()
        reword_recorder.start_episode()
        step = 0
        while True:
            action = agent.act(ob, reward, done)
            reword_recorder.add_value(reward)

            interaction = Interaction(ob, action, None, None, None)
            ob, reward, done, _ = env.step(action)

            interaction = Interaction(interaction.state, interaction.action, ob, reward, done)

            buffer.add_value(interaction)

            if len(buffer) > BATCH_SIZE:
                agent.train(buffer, BATCH_SIZE, ALPHA)

            env.render()

            step += 1

            if done:
                if i % 100 == 0:
                    print('episode {} done took {} steps'.format(i, step))

                reword_recorder.recorde_episode()

                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()

    reword_recorder.show()

