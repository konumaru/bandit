import numpy as np


class Bandit():
    def __init__(self, n_arm, context=None):
        self.n_arm = n_arm
        self.context = context

        self.avg_rewards = np.zeros(n_arm)
        self.pulled_count = 0
        self.is_context = True if context else False

    def update(self):
        pass

    def select_arm(self):
        pass
