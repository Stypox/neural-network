import numpy as np

class QuadraticCost:
	@staticmethod
	def fn(a, y): return 0.5 * np.linalg.norm(a-y)**2
	@staticmethod
	def derivative(z, a, y, f): return (a-y) * f.derivative(z)

class CrossEntropyCost:
	@staticmethod
	def fn(a, y): np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
	@staticmethod
	def derivative(z, a, y, f): return (a-y)