import os
import sys
sys.path.append(os.pardir)

import numpy as np
from bandit import EpsilonGreedy

import matplotlib.pyplot as plt
plt.style.use('ggplot')


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

        data = np.c_[pulled_arms, pulled_rewards]
        return features, data, arm_weights
    else:
        weights = np.random.rand(n_arm)
        pulled_arms = np.random.randint(n_arm, size=n_samples)
        theta = np.random.rand(n_samples)

        pulled_rewards = (theta < weights[pulled_arms]).astype(np.int8)

        data = np.c_[pulled_arms, pulled_rewards]
        return data, weights


def plot_result(result_dict):
    plt.figure()
    plt.xlabel('Select Arm Round')
    plt.ylabel('Average Rewards')
    plt.ylim(0.0, 1.0)
    for key, val in result_dict.items():
        plt.plot(val, label=key)
    plt.legend()
    plt.savefig('./images/main/result.png')


def main():
    n_arm = 5
    n_samples = 1000
    is_contextual = True
    feature_dim = 20

    f_vector, data, weight = generate_data(n_arm, n_samples, is_contextual, feature_dim)

    print('Each Arm Weight Is ', weight)
    print('Max Rewards Arm_id Is ', np.argmax(weight))

    bandit_model = EpsilonGreedy(n_arm)

    batch_size = 10000
    avg_pred_rewads = []
    avg_rand_rewads = []
    result_dict = {'avg_pred_rewads': [],
                   'avg_rand_rewads': []}
    for start in range(0, n_samples, batch_size):
        end = np.minimum(start + batch_size, data.shape[0])

        batch_data = data[start:end]
        # Bandit モデルの更新
        bandit_model.update(data[:end])
        # アームの選択（予測）、選ばれたアームから報酬を計算
        selected_arm = bandit_model.select_arm(end - start)
        is_matched = (batch_data[:, 0] == selected_arm)
        avg_pred_reward = batch_data[is_matched, 1].mean()
        result_dict['avg_pred_rewads'].append(avg_pred_reward)

        # ランダムにアームを選択した場合の報酬を計算
        rand_selected_arms = np.random.choice(np.arange(n_arm), size=(end - start))
        is_matched = (batch_data[:, 0] == rand_selected_arms)
        avg_rand_reward = batch_data[is_matched, 1].mean()
        result_dict['avg_rand_rewads'].append(avg_rand_reward)

    # 辞書型
    plot_result(result_dict)


if __name__ == '__main__':
    main()
