import numpy as np
import pandas as pd

class Bandit():
	def __init__(self, n_arm, context=None):
		self.n_arm = n_arm
		self.context = context

		avg_rewards = np.zeros(n_arm)
		self.is_context = True if context else False

		

