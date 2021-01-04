import gym
import argparse
from gym import wrappers, logger

from src.models import Buffer, CartPoleAgent, Interaction, RewordRecorder


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

    episode_count = 600
    reward = 0
    done = False
    reword_recorder = RewordRecorder()

    BATCH_SIZE = 128 
    GAMMA = 0.9 # qlearning param
    render = False

    actions = {0: 0, 1:0}

    buffer = Buffer(90000)

    for i in range(episode_count):
        ob = env.reset()
        reword_recorder.start_episode()
        step = 0
        while True:
            action = agent.act(ob, reward, done)

            actions[action] += 1

            interaction = Interaction(ob, action, None, None, None)
            ob, reward, done, _ = env.step(action)

            reword_recorder.add_value(reward)

            interaction = Interaction(interaction.state, interaction.action, ob, reward, done)

            buffer.add_value(interaction)

            if len(buffer) > BATCH_SIZE:  
                agent.train(buffer, BATCH_SIZE, GAMMA)

            if render:  env.render()

            step += 1

            if done:
                if i % 50 == 0:
                    print('episode {} done took {} steps'.format(i, step))
                reword_recorder.recorde_episode()
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()

    reword_recorder.show()

    print(actions)