from agent import Agent
from gridWorld import GridWorld


def main():
    env = GridWorld()
    agent = Agent()

    data = [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]

    gamma = 1.0
    alpha = 0.0001


    for k in range(5000):
        done = False
        history = []
        while not done:
            action = agent.select_action()
            (x, y), reward, done = env.step(action)
            history.append((x, y, reward))
        env.reset()

        cum_reward = 0
        for transition in history[::-1]:
            x, y, reward = transition
            data[x][y] += alpha * (cum_reward - data[x][y])
            cum_reward = reward + gamma * cum_reward

    for row in data:
        print(row)
        

if __name__ == '__main__':
    main()
