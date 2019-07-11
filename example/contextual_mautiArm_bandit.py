import os
import sys
sys.path.append(os.pardir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.linear_model import LogisticRegression

from bandit import EpsilonGreedy


def generate_data(n_arm=5, n_samples=10000, is_context=False, feature_dim=5, random_state=None):
    np.random.seed(random_state)

    if is_context:
        # TODO: もっといいfeatureID の与え方ないかな、。。
        feature_id = np.arange(n_samples).astype(np.int64)
        feature_data = np.random.rand(n_samples, feature_dim)
        pulled_arms = np.random.randint(n_arm, size=n_samples)

        # arm_weights = np.random.rand(n_arm, feature_dim)
        arm_weights = np.tile(np.arange(0.1, 1, 1 / n_arm), (feature_dim, 1)).T
        print(arm_weights)
        arm_weights = np.abs(arm_weights - (np.eye(n_arm, feature_dim) / 2))

        rewards = []
        for f_data, arm_id in zip(feature_data, pulled_arms):
            theta = np.dot(f_data, arm_weights[arm_id, :].T) / (feature_dim / 2)
            reward = (theta > np.random.rand()).astype(np.int8)
            rewards.append(reward)

        trial_data = np.c_[feature_id, pulled_arms, rewards]
        return feature_data, trial_data, arm_weights
    else:
        feature_id = np.arange(n_samples)

        arm_weights = np.arange(0.1, 1, 1 / n_arm)
        weights = arm_weights
        pulled_arms = np.random.randint(n_arm, size=n_samples)
        theta = np.random.rand(n_samples)

        rewards = (theta < weights[pulled_arms]).astype(np.int8)

        trial_data = np.c_[feature_id, pulled_arms, rewards]
        return np.zeros(n_samples), trial_data, weights


def plot_result(result_dict, dst_filename):
    plt.figure()
    plt.xlabel('Select Arm Round')
    plt.ylabel('Average Rewards')
    plt.ylim(0.0, 1.0)
    for key, val in result_dict.items():
        plt.plot(val, label=key)
    plt.legend()
    plt.savefig(dst_filename)


def main():
    n_arm = 5
    n_samples = 10000
    is_contextual = True
    feature_dim = 20

    # Generate Data
    feature_data, trial_data, weight = generate_data(n_arm, n_samples, is_contextual, feature_dim)
    print(pd.DataFrame(trial_data).groupby(1)[2].mean())

    # Define Model
    bandit_model = EpsilonGreedy(n_arm)
    contextual_bandit_model = EpsilonGreedy(n_arm)

    # Define Variables
    batch_size = 1000
    result_dict = {'bandit_avg_reward': [], 'contetual_bandit_avg_rewardd': [], 'rand_avg_rewads': []}
    if is_contextual:
        attr = {'solver': 'lbfgs'}
        base_estimator = LogisticRegression
        # TODO: set_context のタイミングでfeature_data をクラスに与える。
        contextual_bandit_model.set_context(feature_data, base_estimator, attr)

    # Train Model
    for start in range(0, n_samples, batch_size):
        end = np.minimum(start + batch_size, trial_data.shape[0])
        print(f'Batch Progress is {end} of {n_samples}')

        batch_data = trial_data[:end]

        # Bandit モデルの更新
        bandit_model.update(trial_data[:end])
        # アームの選択（予測）、選ばれたアームから報酬を計算
        selected_arms = bandit_model.select_arm(batch_data)
        is_observed = (batch_data[:, 1] == selected_arms)
        result_dict['bandit_avg_reward'].append(batch_data[is_observed, 2].mean())

        # Contextual Bandit モデルの更新
        contextual_bandit_model.update(trial_data[:end])
        # アームの選択（予測）、選ばれたアームから報酬を計算
        selected_arms = contextual_bandit_model.select_arm(batch_data)
        is_observed = (batch_data[:, 1] == selected_arms)
        result_dict['contetual_bandit_avg_rewardd'].append(batch_data[is_observed, 2].mean())

        # ランダムにアームを選択した場合の報酬を計算
        rand_selected_arms = np.random.randint(n_arm, size=(end))
        is_observed = (batch_data[:, 1] == rand_selected_arms)
        result_dict['rand_avg_rewads'].append(batch_data[is_observed, 2].mean())

    # 辞書型
    plot_result(result_dict, '../images/main/result.png')


if __name__ == '__main__':
    main()
