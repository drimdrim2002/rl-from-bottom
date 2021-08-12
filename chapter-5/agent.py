import random

random.seed(0)

class Agent ():
    def __init__(self) -> None:
        pass

    def select_action(self):
        coin = random.random()
        if coin < 0.25:
            action = 0
        elif coin < 0.5:
            action = 1
        elif coin < 0.75:
            action = 2
        elif coin < 1:
            action = 3
        else :
            print ("error")
        return action
