class Bandit():
	def __init__(self, n_arm, context=None):
		self.n_arm = n_arm
		self.context = context

		self.is_context = True if context else False
		

