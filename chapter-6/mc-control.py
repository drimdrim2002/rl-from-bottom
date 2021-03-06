from gridWorld import GridWorld
from qAgent import QAgent


def main() :
    env = GridWorld()
    agent = QAgent()

    for episode in range (1000):
        done = False
        history = []

        state = env.reset()
        while not done:
            action = agent.select_action(state)
            s_prime , reward, done = env.step(action)
            history.append((state, action, reward, s_prime))
            state = s_prime
        agent.update_table_mc(history)
        agent.anneal_eps()
    agent.show_table()


if __name__ == "__main__" :
    main()
