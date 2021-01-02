import gym
# import vizdoom
env = gym.make('VizdoomBasic-v0')

# use like a normal Gym environment
state = env.reset()
while True:
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()
    if done: break

env.close()