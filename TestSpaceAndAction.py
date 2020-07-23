import gym
env = gym.make('CartPole-v0')
print(env.action_space)
print(env.observation_space)
print(env.reward_range)
print(env.metadata)