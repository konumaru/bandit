import numpy as np


class BaseBandit():

    def __init__(self, n_arm):
        self.n_arm = n_arm
        self.selected_counts = np.zeros(n_arm)
        self.each_arm_probability = np.zeros(n_arm)

        self.is_context = False


class EpsilonGreedy(BaseBandit):

    def __init__(self, n_arm, epsilon=0.2):
        super().__init__(n_arm)
        self.epsilon = epsilon

    # MEMO: Thompson Sampling の場合、estimator_shapeを２次元にすればよい
    def _set_estimators(self, model_class, attrs, estimator_shape):
        def f(x): return model_class(**attrs)
        vf = np.vectorize(f)
        estimators = vf(np.empty(estimator_shape))
        return estimators

    def set_context(self, feature_data, estimator, attrs):
        self.is_context = True
        self.feature_data = feature_data
        self.estimators = self._set_estimators(estimator, attrs, (self.n_arm))

    def update(self, trial_data):
        if self.is_context:
            self._contextual_update(trial_data)
        else:
            self._normal_update(trial_data)

    def _normal_update(self, trial_data):
        self.each_arm_probability = np.array([trial_data[(trial_data[:, 1] == arm_id), 2].mean()
                                              for arm_id in range(self.n_arm)])

    def _contextual_update(self, trial_data):
        # アームごとにestimator を学習
        for arm_id in range(self.n_arm):
            is_matched_arm = (trial_data[:, 1] == arm_id)
            # TODO: もっとうまく書けないのかな。
            f = self.feature_data
            f_id = f[:, 0]
            f_data = f[:, 1:]
            mated_f_idx = np.array([np.where(f_id == i)[0] for i in trial_data[is_matched_arm, 0]]).flatten()

            X = f_data[mated_f_idx]
            y = trial_data[is_matched_arm, 2]

            self.estimators[arm_id].fit(X, y)

    def select_arm(self, trial_data):
        pull_count = trial_data.shape[0]
        # 最適なアームを選択
        if self.is_context:
            selected_arm = self._get_contextual_selected_arm(trial_data)
        else:
            optimal_arm_id = self.each_arm_probability.argmax()
            selected_arm = np.full(pull_count, fill_value=optimal_arm_id)

        # 探索フラグを指定、探索アームを割り当て
        explore_flag = np.random.rand(pull_count) < self.epsilon
        explore_count = sum(explore_flag)
        selected_arm[explore_flag] = np.random.choice(np.arange(self.n_arm), size=explore_count)
        return selected_arm

    def _get_contextual_selected_arm(self, trial_data):
        pulling_feature_id = trial_data[:, 0]
        # TODO: もっとうまく書けないのかな。
        f = self.feature_data
        f_id = f[:, 0]
        f_data = f[:, 1:]
        mated_f_idx = np.array([np.where(f_id == i)[0] for i in pulling_feature_id]).flatten()
        X = f_data[mated_f_idx]

        each_arm_proba = np.array([e.predict_proba(X)[:, 1] for e in self.estimators]).T
        return each_arm_proba.argmax(axis=1)
