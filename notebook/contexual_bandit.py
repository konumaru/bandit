import numpy as np
from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression


class ZeroPredictor():

    def predict_proba(self, X):
        return np.c_[np.ones((X.shape[0], 1)),  np.zeros((X.shape[0], 1))]


class OnePredictor():

    def predict_proba(self, X):
        return np.c_[np.zeros((X.shape[0], 1)), np.ones((X.shape[0], 1))]


class ContextualBandit():

    def __init__(self, base_model, n_arm,  n_estimator=100):
        self.n_arm = n_arm
        self.base_model = base_model
        self.n_estimator = n_estimator
        self.estimators = np.array(
            [[base_model for _ in range(n_estimator)] for _ in range(n_arm)])
        self.smpl_count_each_arm = np.zeros(n_arm, dtype=np.int64)

    def fit(self, X, chosen_arm, y):
        for arm_id in range(self.n_arm):
            matched_X = X[(chosen_arm == arm_id)]
            matched_y = y[(chosen_arm == arm_id)]

            Parallel(n_jobs=-1, verbose=0, require="sharedmem")(
                [delayed(self._fit_single)(arm_id, estimator_idx, matched_X, matched_y)
                 for estimator_idx in range(self.n_estimator)])

            self.smpl_count_each_arm[arm_id] += matched_X.shape[0]

    def _fit_single(self, arm_id, estimator_idx, X, y):
        _X, _y = self._bootstrapped_sampling(X, y)
        if _y.sum() == 0:
            self.estimators[arm_id, estimator_idx] = ZeroPredictor()
            return None
        elif _y.sum() == _y.shape[0]:
            self.estimators[arm_id, estimator_idx] = OnePredictor()
            return None
        self.estimators[arm_id, estimator_idx].fit(_X, _y)

    def _bootstrapped_sampling(self, X, y, sample_rate=0.8):
        data_size = X.shape[0]
        bootstrapped_idx = np.random.randint(
            0, data_size, int(data_size*sample_rate))
        return X[bootstrapped_idx], y[bootstrapped_idx]

    def _thompson_sampling(self, smpl):
        smpl_avg, smpl_std = smpl.mean(), smpl.std()
        return np.random.normal(smpl_avg, smpl_std)

    def _get_proba_with_thompson_sampling(self, arm_id, X):

        def _single_predict_proba(arm_id, estimator_idx, X):
            return self.estimators[arm_id, estimator_idx].predict_proba(X)[:, 1]

        proba_result = Parallel(n_jobs=-1)(
            [delayed(_single_predict_proba)(arm_id, estimator_idx, X)
             for estimator_idx in range(self.n_estimator)])

        proba_mean = np.mean(proba_result, axis=0)
        proba_std = np.std(proba_result, axis=0)

        return [np.random.normal(p, s) for p, s in zip(proba_mean, proba_std)]

    def predict(self, X):
        proba_each_users = np.zeros((self.n_arm, X.shape[0]))

        for arm_id in range(self.n_arm):
            proba_each_users[arm_id, :] = \
                self._get_proba_with_thompson_sampling(arm_id, X)

        return np.argmax(proba_each_users, axis=0)
