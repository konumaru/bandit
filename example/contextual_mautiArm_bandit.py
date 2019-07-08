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
        features = np.random.rand(n_samples, feature_dim)
        arm_weights = np.random.rand(n_arm, feature_dim)
        arm_weights = arm_weights / arm_weights.sum(axis=0)

        pulled_arms = np.random.randint(n_arm, size=n_samples)

        pulled_rewards = []
        for i, arm_id in enumerate(pulled_arms):
            theta = np.dot(features[i], arm_weights[arm_id].T)
            noise = np.dot(np.random.rand(feature_dim), arm_weights[arm_id].T)
            rewards = (theta > noise).astype(np.int8)
            pulled_rewards.append(rewards)

        rewards = np.c_[pulled_arms, pulled_rewards]
        return features, rewards, arm_weights
    else:
        weights = np.random.rand(n_arm)
        pulled_arms = np.random.randint(n_arm, size=n_samples)
        theta = np.random.rand(n_samples)

        pulled_rewards = (theta < weights[pulled_arms]).astype(np.int8)

        rewards = np.c_[pulled_arms, pulled_rewards]
        return None, rewards, weights


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
    f_vector, rewards, weight = generate_data(n_arm, n_samples, is_contextual, feature_dim)

    bandit_model = EpsilonGreedy(n_arm)

    batch_size = 1000
    result_dict = {'avg_pred_rewads': [], 'avg_rand_rewads': []}

    for start in range(0, n_samples, batch_size):
        end = np.minimum(start + batch_size, rewards.shape[0])
        print(f'Batch Progress Is {end} of {n_samples}')

        batch_data = rewards[start:end]
        batch_pulled_arms = batch_data[:, 0]
        batch_rewards = batch_data[:, 1]

        # Bandit モデルの更新
        bandit_model.update(rewards[:end])
        # アームの選択（予測）、選ばれたアームから報酬を計算
        selected_arms = bandit_model.select_arm(end - start)
        is_observed = (batch_pulled_arms == selected_arms)
        avg_pred_reward = batch_rewards[is_observed].mean()
        result_dict['avg_pred_rewads'].append(avg_pred_reward)

        # ランダムにアームを選択した場合の報酬を計算
        rand_selected_arms = np.random.randint(n_arm, size=(end - start))
        is_observed = (batch_pulled_arms == rand_selected_arms)
        avg_rand_reward = batch_rewards[is_observed].mean()
        result_dict['avg_rand_rewads'].append(avg_rand_reward)

    # 辞書型
    plot_result(result_dict, '../images/main/result.png')


if __name__ == '__main__':
    main()
