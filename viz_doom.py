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

    env = gym.make(args.env_id, depth=True, labels=True, position=True, health=True)

    agent = VizDoomAgent(env)

    episode_count = 5
    reward = 0
    done = False
    reword_recorder = RewordRecorder()

    BATCH_SIZE = 30
    GAMMA = 0.9 # qlearning param
    render = False

    PATH = 'saved_nets/viz_doom_cnn.pt'

    actions = {0: 0, 1:0, 2:0}

    buffer = Buffer(20000)

    skip_frame = 10

    for i in range(episode_count):
        ob = env.reset()
        reword_recorder.start_episode()
        step = 0
        while True:

            action = agent.act(agent.preprocess(ob), reward, done)

            if step % skip_frame == 0:
                interaction = Interaction(agent.preprocess(ob), action, None, None, None)

            if step % 50 == 0:
                print(step)

            if step % skip_frame == 0:
                ob, reward, done, _ = env.step(action)
                interaction = Interaction(interaction.state, interaction.action, agent.preprocess(ob), reward, done)
            else:
                _, reward, _, _ = env.step(action)

            
            reword_recorder.add_value(reward)

            actions[action] += 1

            if step % skip_frame == 0:
                buffer.add_value(interaction)

                if len(buffer) > BATCH_SIZE:  
                    agent.train(buffer, BATCH_SIZE, GAMMA)

            if render:  env.render()

            step += 1

            if done:
                # if i % 10 == 0:
                reword_recorder.recorde_episode()
                print('episode {} done took {} steps and reward {} '.format(i, step, reword_recorder.rewards[-1]))
                break

        # env.render()
        # input('next?')

    env.close()
    reword_recorder.show()

    agent.save(PATH)



