import random
import numpy as np

random.seed(0)


class QAgent():
    def __init__(self):
        self.q_table = np.zeros((5, 7, 4))
        self.eps = 0.9
        self.alpha = 0.01

    def select_action(self, state):
        x, y = state
        coin = random.random()

        if coin < self.eps:  # random
            action = random.randint(0, 3)
        else:
            action_val = self.q_table[x, y, :]
            action = np.argmax(action_val)
        return action

    def update_table(self, history):  # history는 모든 에피소드의 결과를 받아 q 테이블의 값을 업데이트 한다.
        cum_reward = 0
        for transition in history[::-1]:
            state, action, reward, s_prime = transition
            x, y = state
            selft.q_table[x, y, action] += self.alpha * (cum_reward - selft.q_table[x, y, action])
            cum_reward += reward

    def anneal_eps(self):
        self.eps -= 0.03
        self.eps = max(self.eps, 0.1)

    def show_table(self):
        q_list = self.q_table.tolist()
        data = np.zeros((5,7))
        for row_idx in range(len(q_list))
            row = q_list[row_idx]
            for col_idx in range(len(row)):
                col = row[col_idx]
                action = np.argmax(col)
                data[row_idx, col_idx] = action
        print (data)