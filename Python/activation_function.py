import numpy as np

class Sigmoid:
	@staticmethod
	def fn(z): return 1.0/(1.0+np.exp(-z))
	@staticmethod
	def derivative(z): return Sigmoid.fn(z)*(1-Sigmoid.fn(z))