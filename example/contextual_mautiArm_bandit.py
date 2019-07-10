import os
import sys
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.linear_model import LogisticRegression

from bandit import EpsilonGreedy


def generate_data(n_arm=5, n_samples=10000, is_context=False, feature_dim=5, random_state=42):
    np.random.seed(random_state)

    if is_context:
        # TODO: もっといいfeatureID の与え方ないかな、。。
        feature_id = np.arange(n_samples)
        feature_data = np.c_[feature_id, np.random.rand(n_samples, feature_dim)]
        arm_weights = np.random.rand(n_arm, feature_dim) / 2
        arm_weights = arm_weights / arm_weights.sum(axis=0)

        pulled_arms = np.random.randint(n_arm, size=n_samples)

        rewards = []
        for i, arm_id in enumerate(pulled_arms):
            theta = np.dot(feature_data[i][1:], arm_weights[arm_id].T)
            noise = np.dot(np.random.rand(feature_dim), arm_weights[arm_id].T)
            reward = (theta > noise).astype(np.int8)
            rewards.append(reward)

        trial_data = np.c_[feature_id, pulled_arms, rewards]
        return feature_data, trial_data, arm_weights
    else:
        feature_id = np.arange(n_samples)
        weights = np.random.rand(n_arm) / 2
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
    is_contextual = False
    feature_dim = 20

    # Generate Data
    feature_data, trial_data, weight = generate_data(n_arm, n_samples, is_contextual, feature_dim)

    # Define Variables
    batch_size = 1000
    result_dict = {'avg_pred_rewads': [], 'avg_contetual_pred_reward': [], 'avg_rand_rewads': []}
    bandit_model = EpsilonGreedy(n_arm)
    contextual_bandit_model = EpsilonGreedy(n_arm)

    # bandit_model にコンテキストを与える
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
        avg_pred_reward = batch_data[is_observed, 2].mean()
        result_dict['avg_pred_rewads'].append(avg_pred_reward)

        # Contextual Bandit モデルの更新
        contextual_bandit_model.update(trial_data[:end])
        # アームの選択（予測）、選ばれたアームから報酬を計算
        selected_arms = contextual_bandit_model.select_arm(batch_data)
        is_observed = (batch_data[:, 1] == selected_arms)
        avg_contetual_pred_reward = batch_data[is_observed, 2].mean()
        result_dict['avg_contetual_pred_reward'].append(avg_contetual_pred_reward)

        # ランダムにアームを選択した場合の報酬を計算
        rand_selected_arms = np.random.randint(n_arm, size=(end))
        is_observed = (batch_data[:, 1] == rand_selected_arms)
        avg_rand_reward = batch_data[is_observed, 2].mean()
        result_dict['avg_rand_rewads'].append(avg_rand_reward)

    # 辞書型
    plot_result(result_dict, '../images/main/result.png')


if __name__ == '__main__':
    main()
