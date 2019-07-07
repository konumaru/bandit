import numpy as np
from bandit import EpsilonGreedy

import matplotlib.pyplot as plt
plt.style.use('ggplot')


def generate_data(n_arm=5, n_samples=10000, random_state=42):
    np.random.seed(random_state)

    n_noise = np.random.rand(n_samples)
    weights = np.random.rand(n_arm)

    pulled_arms = np.random.randint(n_arm, size=n_samples)
    pulled_rewards = (n_noise < weights[pulled_arms]).astype(np.int8)

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
    n_samples = 1000000
    data, weight = generate_data(n_arm, n_samples)

    print('Each Arm Weight Is ', weight)
    print('Max Rewards Arm_id Is ', np.argmax(weight))
    print('Max Average Rewards Score Is ', weight[np.argmax(weight)])

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
