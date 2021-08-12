import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])

learning_rate = 0.85
dis = 0.99
num_episodes = 2000

# create lists to contain total rewards and steps per episodes
rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    # Q-table learning algorithm
    while not done:
        # choose an action by greedily (with noise) picking from Q table
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))

        # get new state, reward from environment
        new_state, reward, done, _ = env.step(action)

        # update Q-table with new knowledge using decay rate
