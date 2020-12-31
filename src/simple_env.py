import gym
from gym import spaces
space = spaces.Discrete(8)

env = gym.make('CartPole-v1')
print(env.observation_space)
print(env.action_space)

for i_episode in range(20):
    observation = env.reset()
    print(observation)
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(1)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
