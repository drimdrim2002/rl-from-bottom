from gridWorld import GridWorld
from qAgent import QAgent


def main() :
    env = GridWorld()
    agent = QAgent()

    for episode in range (1000):
        done = False
        state = env.reset()
        while not done:
            action = agent.select_action(state)
            state_prime , reward, done = env.step(action)
            agent.update_table_sarsa((state, action, reward, state_prime))
            state = state_prime
        agent.anneal_eps()
    agent.show_table()


if __name__ == "__main__" :
    main()
