import numpy as np
from bandit import Bandit


def generate_data(n_arm=5, n_samples=10000):
    np.random.seed(n_arm)

    n_noise = np.random.rand(n_samples)
    weights = np.random.rand(n_arm) / 2

    pulled_arm = np.random.randint(n_arm, size=n_samples)
    rewards = (n_noise < weights[pulled_arm]).astype(np.int8)

    return pulled_arm, rewards, weights


def main():
    bandit_model = Bandit(n_arm=3)

    n_arm = 5
    n_samples = 10000
    pulled_arm, rewards, weight = generate_data(n_arm, n_samples)

    print(weight)

    bandit_model = Bandit(n_arm)
    batch_size = 100

    for i in range(0, n_samples + batch_size, batch_size):
        start = i
        end = i + batch_size

        bandit_model.update()


if __name__ == '__main__':
    main()
