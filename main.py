import os

import numpy as np
import pandas as pd

from bandit import Bandit

def generate_data(n_arm=5, n_samples=10000):
	np.random.seed(n_arm)

	n_noise = np.random.rand(n_samples)
	weights = np.random.rand(n_arm) / 2

	pulled_arm = np.random.randint(n_arm, size=n_samples)
	rewards = (n_noise < weights[pulled_arm]).astype(np.int8)

	return pulled_arm, rewards, weights

def train_bandit():
	pass


def main():
	bandit_model = Bandit(n_arm=3)

	n_arm=5
	n_samples=10000	
	pulled_arm, rewards, weight = generate_data()

	train_bandit()




if __name__=='__main__':
	main()
