import numpy as np
import random

class Network:
	def __init__(self, sizes, costFunction, activationFunction):
		self.numLayers = len(sizes)
		self.sizes = sizes
		self.costFunction = costFunction
		self.activationFunction = activationFunction

		self.biases = [np.random.normal(size=(ySize, 1), scale=1.0)
			for ySize in sizes[1:]]
		self.weights = [np.random.normal(size=(ySize, yFromSize), scale=1.0/np.sqrt(yFromSize))
			for ySize, yFromSize in zip(sizes[1:], sizes[:-1])]

		for b in self.biases: print("Bias:", b.shape)
		for w in self.weights: print("Weight:", w.shape)

	
	def feedforward(self, x):
		a = x
		for b, w in zip(self.biases, self.weights):
			a = self.activationFunction.fn(np.dot(w, a) + b)
		return a
	

	def backpropagation(self, x, y):
		# feedforward
		As = [x]
		zs = []
		for b, w in zip(self.biases, self.weights):
			zs.append(np.dot(w, As[-1]) + b)
			As.append(self.activationFunction.fn(zs[-1]))
		
		# output layer backpropagation
		bNablas = [self.costFunction.derivative(zs[-1], As[-1], y, self.activationFunction)] # == error for nodes
		wNablas = [np.dot(bNablas[0], np.transpose(As[-2]))]

		# backpropagation
		for l in range(1, self.numLayers-1):
			bNablas.insert(0, np.dot(np.transpose(self.weights[-l]), bNablas[0]) * self.activationFunction.derivative(zs[-l-1]))
			wNablas.insert(0, np.dot(bNablas[0], np.transpose(As[-l-2])))

		return bNablas, wNablas

	def SGDMiniBatch(self, samples, eta, weightDecayFactor, momentumCoefficient):
		bNablas = [np.zeros(b.shape) for b in self.biases]
		wNablas = [np.zeros(w.shape) for w in self.weights]

		for x, y in samples:
			bNablasDelta, wNablasDelta = self.backpropagation(x, y)
			bNablas = [bns+bnsd for bns, bnsd in zip(bNablas, bNablasDelta)]
			wNablas = [wns+wnsd for wns, wnsd in zip(wNablas, wNablasDelta)]
		
		etaScaled = eta/len(samples)
		self.bVelocities = [momentumCoefficient*bVelocity - etaScaled*bNabla
			for bVelocity, bNabla in zip(self.bVelocities, bNablas)]
		self.wVelocities = [momentumCoefficient*wVelocity - etaScaled*wNabla
			for wVelocity, wNabla in zip(self.wVelocities, wNablas)]

		self.biases = [b + bVelocity
			for b, bVelocity in zip(self.biases, self.bVelocities)]
		self.weights = [weightDecayFactor*w + wVelocity
			for w, wVelocity in zip(self.weights, self.wVelocities)]
	
	def SGDEpoch(self, samples, miniBatchSize, eta, regularizationParameter, momentumCoefficient):
		random.shuffle(samples)

		# reset velocities
		self.bVelocities = [np.zeros(b.shape) for b in self.biases]
		self.wVelocities = [np.zeros(w.shape) for w in self.weights]

		weightDecayFactor = (1.0 - eta*regularizationParameter/len(samples)) # TODO maybe remove len(samples)
		for start in range(0, len(samples), miniBatchSize):
			self.SGDMiniBatch(samples[start:start+miniBatchSize], eta, weightDecayFactor, momentumCoefficient)
	
	def SGD(self, trainingSamples, epochs, miniBatchSize, eta, regularizationParameter, momentumCoefficient, testSamples=None):
		samples = list(trainingSamples)
		tests = None if testSamples is None else list(testSamples)

		for e in range(epochs):
			self.SGDEpoch(samples, miniBatchSize, eta, regularizationParameter, momentumCoefficient)

			if testSamples is None:
				print("Epoch", e+1)
			else:
				print("Epoch ", e+1, ": ", self.evaluate(tests), " / ", len(tests), sep="")
	

	def evaluate(self, testSamples):
		correct = 0
		for x, y in testSamples:
			a = self.feedforward(x)
			correct += int(y == np.argmax(a))
		return correct