import numpy as np


class BaseBandit():

    def __init__(self, n_arm, context=None):
        self.n_arm = n_arm
        self.context = context

        self.avg_rewards = np.zeros(n_arm)
        self.is_context = True if context else False


class EpsilonGreedy(BaseBandit):

    def __init__(self, n_arm, context=None, epsilon=0.2):
        super().__init__(n_arm, context=context)
        self.epsilon = epsilon

    def update(self, data):
        self.avg_rewards = np.array([data[(data[:, 0] == arm_id), 1].mean()
                                     for arm_id in range(self.n_arm)])

    def select_arm(self, pull_count):
        # 最適なアームを選択
        optimal_arm_id = self.avg_rewards.argmax()
        selected_arm = np.full(pull_count, fill_value=optimal_arm_id)

        # 探索フラグを指定、探索アームを割り当て
        explore_flag = np.random.rand(pull_count) < self.epsilon
        explore_count = sum(explore_flag)
        selected_arm[explore_flag] = np.random.choice(np.arange(self.n_arm), size=explore_count)
        return selected_arm
