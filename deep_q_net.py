# partie 2 Deep Q-network sur CartPole


import argparse

import gym
from gym import wrappers, logger


import matplotlib.pyplot as plt
import numpy as np


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
        self.mean_rewards = []

    def start_episode(self):
        self.c = 0
        self.r_sum = 0

    def add_value(self, reward):
        self.c += 1
        self.r_sum += reward

    def recorde_episode(self):
        self.mean_rewards.append(self.r_sum/self.c)

    def show(self):
        plt.plot(self.mean_rewards)
        plt.ylabel('avg rewards par episode')
        plt.xlabel('episodes')
        plt.show()


# 2.2.3
class Buffer():
    """
        a 2d generale buffer.

    """
    def __init__(self, buffer_shape: tuple[int, int], dtype) -> None:

        assert len(buffer_shape) == 2

        self.buffer_shape = buffer_shape
        self.dtype = dtype
        self.buffer = np.empty(shape=buffer_shape, dtype=dtype)
        self.cs = 0 # current size


    def get_buffer(self):
        return self.buffer[:self.cs]

    """
        xs = array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
        shift(xs, 3)
        array([ nan,  nan,  nan,   0.,   1.,   2.,   3.,   4.,   5.,   6.])
        shift(xs, -3)
        array([  3.,   4.,   5.,   6.,   7.,   8.,   9.,  nan,  nan,  nan])
    """
    def shift(self, arr, num, fill_value=np.nan):
        if num >= 0:
            return np.concatenate((np.full(num, fill_value), arr[:-num]))
        else:
            return np.concatenate((arr[-num:], np.full(-num, fill_value)))

    def add_value(self, value):

        assert self.buffer_shape[1] == len(value)

        if self.cs >= self.buffer_shape[0]:
            self.cs = self.buffer_shape[0]
            # shift 1 left so remove buffer[0]
            self.buffer = self.shift(self.buffer, -1, np.empty(shape=(self.buffer_shape[1],), dtype=self.dtype))
            self.buffer[-1, : ] = value
        else:
            self.buffer[self.cs, :] = value
            self.cs += 1



if __name__ == '__main__':
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
    agent = RandomAgent(env.action_space)

    episode_count = 1000
    reward = 0
    done = False
    reword_recorder = RewordRecorder()
    for i in range(episode_count):
        ob = env.reset()
        reword_recorder.start_episode()
        while True:
            action = agent.act(ob, reward, done)
            reword_recorder.add_value(reward=reward)
            ob, reward, done, _ = env.step(action)
            # env.render()
            if done:
                reword_recorder.recorde_episode()
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()

    reword_recorder.show()