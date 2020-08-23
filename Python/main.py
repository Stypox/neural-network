from network import Network
from cost_function import CrossEntropyCost
from activation_function import Sigmoid
import numpy as np
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = Network([784, 30, 10], CrossEntropyCost, Sigmoid)
net.SGD(training_data, 30, 10, 0.1, 5.0, 0.2, test_data)