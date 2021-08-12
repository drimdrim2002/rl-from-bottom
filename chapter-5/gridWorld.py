class GridWorld():
    def __init__(self) -> None:
        self.x = 0
        self.y = 0

    def move_right(self):
        if self.y < 3:
            self.y += 1

    def move_left(self):
        if self.y > 0:
            self.y -= 1

    def move_up(self):
        if self.x > 0:
            self.x -= 1

    def move_down(self):
        if self.x < 3:
            self.x += 1

    def is_done(self):
        if self.x == 3 and self.y == 3:
            return True
        else:
            return False

    def step(self, action):
        if action == 0:
            self.move_right()
        elif action == 1:
            self.move_left()
        elif action == 2:
            self.move_up()
        elif action == 3:
            self.move_down()

        reward = -1
        done = self.is_done()
        return (self.x, self.y), reward, done

    def get_state(self):
        return (self.x, self.y)

    def reset(self):
        self.x = 0
        self.y = 0
        return (self.x, self.y)
