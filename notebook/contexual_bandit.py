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
            [[base_model for i in range(n_estimator)] for _ in range(n_arm)])
        self.smpl_count_each_arm = np.zeros(n_arm, dtype=np.int64)

    def fit(self, X, chosen_arm, y):
        for arm_id in range(self.n_arm):
            matched_X = X[(chosen_arm == arm_id)]
            matched_y = y[(chosen_arm == arm_id)]

            self.estimators[arm_id, :] = Parallel(n_jobs=-1)(
                [delayed(self._bagging)(estimeator, matched_X, matched_y
                                        )for estimeator in self.estimators[arm_id, :]])

    def _bagging(self, model, _X, _y):
        _X, _y = self._bootstrapped_sampling(X, y)
        if np.unique(_y)[0] == 0:
            return ZeroPredictor()
        elif np.unique(_y)[0] == 1:
            return OnePredictor()
        
        model.fit(_X, _y)
        return model

    def _bootstrapped_sampling(self, X, y, sample_rate=0.8):
        data_size = X.shape[0]
        bootstrapped_idx = np.random.randint(
            0, data_size, int(data_size*sample_rate))
        return X[bootstrapped_idx], y[bootstrapped_idx]

    def _thompson_sampling(self, smpl):
        smpl_avg, smpl_std = smpl.mean(), smpl.std()
        return np.random.normal(smpl_avg, smpl_std)

    def predict(self, X):
        proba_by_arms = np.zeros((X.shape[0], self.n_arm))

        def predict_proba(estimators, X, arm_id):
            if self.smpl_count_each_arm[arm_id] < 100:
                return np.random.beta(2, 75, size=X.shape[0])
            else:
                proba = np.array([e.predict_proba(X)[:, 1]
                                  for e in estimators]).T
                proba = [self._thompson_sampling(p) for p in proba]
                return np.array(proba)

        proba_by_arms = Parallel(n_jobs=-1)([delayed(predict_proba)(
            self.estimators[arm_id], X, arm_id) for arm_id in range(self.n_arm)])

        return np.argmax(proba_by_arms, axis=0)
