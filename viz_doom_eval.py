import gym
import vizdoomgym
import numpy as np

from src.models import evaluate_agent, VizDoomAgent, Interaction, RewordRecorder


import gym
import argparse
from gym import wrappers, logger

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='VizdoomBasic-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id, depth=True, labels=True, position=True, health=True)

    agent = VizDoomAgent(env)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    PATH = 'saved_nets/viz_doom_best.pt'
    # env = wrappers.Monitor(env, directory=outdir, force=True)
    agent.load(PATH)

    episode_count = 250

    reword_recorder, actions = evaluate_agent(env, agent, episode_count, render=False)


    # Close the env and write monitor result info to disk
    env.close()
    reword_recorder.show()
