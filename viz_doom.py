import gym
import vizdoomgym
import numpy as np
from skimage.viewer import ImageViewer

from src.models import Buffer, VizDoomAgent, Interaction, RewordRecorder
from src import models


import gym
import argparse
from gym import wrappers, logger



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='VizdoomBasic-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    agent = VizDoomAgent(env)

    episode_count = 1000
    reward = 0
    done = False
    reword_recorder = RewordRecorder()

    BATCH_SIZE = 128
    GAMMA = 0.2 # qlearning param
    render = False

    actions = {0: 0, 1:0}

    buffer = Buffer(90000)

    for i in range(episode_count):
        ob = env.reset()
        reword_recorder.start_episode()
        step = 0
        done = False
        while not done:

            action = agent.act(agent.preprocess(ob), reward, done)
            interaction = Interaction(ob, action, None, None, None)
            ob, reward, done, _ = env.step(action)
            interaction = Interaction(interaction.state, interaction.action, agent.preprocess(ob), reward, done)

            reword_recorder.add_value(reward)


            buffer.add_value(interaction)

            if len(buffer) > BATCH_SIZE:  
                agent.train(buffer, BATCH_SIZE, GAMMA)

            if render:  env.render()

            step += 1

            if done:
                if i % 500 == 0:
                    print('episode {} done took {} steps'.format(i, step))
                reword_recorder.recorde_episode()
                break

        # env.render()
        # input('next?')

    env.close()

