import gym
import vizdoomgym
import numpy as np
from skimage.viewer import ImageViewer

from src.models import train_agent, VizDoomAgent, Interaction, RewordRecorder
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

    config = {
        'gamma': 0.9,
        'skip_frame': 4,
        'batch_size': 64,
        'save_model': True,
        'episode_count': 200,
        'buffer_size': 20000,
        'best_path': 'saved_nets/viz_doom_best.pt',
        # 'last_path': 'saved_nets/viz_doom_last.pt',
    }

    reword_recorder, actions = train_agent(env, agent, config)

    env.close()
    reword_recorder.show()




