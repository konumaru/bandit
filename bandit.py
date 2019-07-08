import numpy as np


class BaseBandit():

    def __init__(self, n_arm):
        self.n_arm = n_arm
        self.selected_counts = np.zeros(n_arm)
        self.each_arm_probability = np.zeros(n_arm)


class EpsilonGreedy(BaseBandit):

    def __init__(self, n_arm, epsilon=0.2):
        super().__init__(n_arm)
        self.epsilon = epsilon

    def _set_estimators(self, model_class, attrs):
        def f(x): return model_class(**attrs)
        vf = np.vectorize(f)
        estimators = vf(np.empty((self.n_arm)))
        return estimators

    def set_context(self, f_vector, estimator, attrs):
        self.is_context = True
        self.f_vector = f_vector
        self.estimators = self._set_estimators(estimator, attrs)

    def update(self, data):
        self.each_arm_probability = np.array([data[(data[:, 0] == arm_id), 1].mean()
                                              for arm_id in range(self.n_arm)])
        # 文脈付きの場合
        if self.is_context:
            # 誰が何を引き、辺りかハズレかのデータを取得
            feature_id = data[:, 0]
            pulled_arm = data[:, 1]
            rewards = data[:, 2]
            # アームごとにestimator を学習
            for arm_id in range(self.n_arm):
                self.estimators[arm_id].fit(self.f_vector[feature_id], rewards)

    # 文脈付きの場合の予測処理
    def select_arm(self, pull_count):
        # 最適なアームを選択
        optimal_arm_id = self.each_arm_probability.argmax()
        selected_arm = np.full(pull_count, fill_value=optimal_arm_id)

        # 探索フラグを指定、探索アームを割り当て
        explore_flag = np.random.rand(pull_count) < self.epsilon
        explore_count = sum(explore_flag)
        selected_arm[explore_flag] = np.random.choice(np.arange(self.n_arm), size=explore_count)
        return selected_arm
