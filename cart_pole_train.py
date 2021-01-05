import gym
import argparse
from gym import wrappers, logger

from src.models import Buffer, CartPoleAgent, Interaction, RewordRecorder, train_agent

import math

if __name__ == "__main__":
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
    agent = CartPoleAgent(env)


    config = {
        'gamma': 0.9,
        'skip_frame': 1,
        'batch_size': 128,
        'save_model': True,
        'episode_count': 1000,
        'buffer_size': 90000,
        'best_path': 'saved_nets/cart_pole_best.pt',
        # 'last_path': 'saved_nets/cart_pole_last.pt',
    }

    reword_recorder, actions = train_agent(env, agent, config)

    env.close()

    reword_recorder.show()
