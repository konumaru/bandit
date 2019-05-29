import numpy as np
from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression

np.random.seed(42)


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

            self.estimators[arm_id] = Parallel(n_jobs=-1)(
                [delayed(self._bagging)(self.base_model, matched_X, matched_y)
                 for estimator in range(self.n_estimator)]
            )

            self.smpl_count_each_arm[arm_id] += matched_X.shape[0]

    def _bagging(self, model, X, y):
        _X, _y = self._bootstrapped_sampling(X, y)
        if _y.sum() == 0:
            return ZeroPredictor()
        elif _y.sum() == _y.shape[0]:
            return OnePredictor()
        else:
            return model.fit(_X, _y)

    def _bootstrapped_sampling(self, X, y, sample_rate=0.8):
        data_size = X.shape[0]
        bootstrapped_idx = np.random.randint(
            0, data_size, int(data_size*sample_rate))
        return X[bootstrapped_idx], y[bootstrapped_idx]

    def _thompson_sampling(self, smpl):
        smpl_avg, smpl_std = smpl.mean(), smpl.std()
        return np.random.normal(smpl_avg, smpl_std)

    def _predict_proba_with_thompson_sampling(self, arm_id, X):

        def _single_predict_proba(model, X):
            return model.predict_proba(X)[:, 1]

        proba_result = Parallel(n_jobs=-1)([delayed(_single_predict_proba)(
            estimator, X) for estimator in self.estimators[arm_id]])

        proba_mean = np.mean(proba_result, axis=0)
        proba_std = np.std(proba_result, axis=0)

        return [np.random.normal(p, s) for p, s in zip(proba_mean, proba_std)]

    def predict(self, X):
        proba_each_users = np.zeros((self.n_arm, X.shape[0]))

        def predict_proba(estimators, X, arm_id):
            proba = np.array([e.predict_proba(X)[:, 1] for e in estimators]).T
            proba = [self._thompson_sampling(p) for p in proba]
            return np.array(proba)

        for arm_id in range(self.n_arm):
            proba_each_users[arm_id, :] = self._predict_proba_with_thompson_sampling(
                arm_id, X)

        return np.argmax(proba_each_users, axis=0)
