import numpy as np


class Neuron(object):
    def __init__(self, bias):
        self.bias = bias
        self.weights = []
        self.last_inputs = []
        self.last_result = 0

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self):
        x = np.dot(self.last_inputs, self.weights) + self.bias
        if x <= 0:
            return 0
        return 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self):
        return self.last_result * (1 - self.last_result)

    def linear(self, x):
        return x

    def linear_derivative(self):
        return 1

    def mse_error(self, target):
        return 0.5 * (target - self.last_result) ** 2

    def mse_error_derivative(self, target):
        return -(target - self.last_result)

    def calculate(self, inputs, activation_method):
        self.last_inputs = inputs
        self.last_result = self.activation(np.dot(inputs, self.weights) + self.bias, activation_method)
        return self.last_result

    def get_last_input_by_index(self, index):
        return self.last_inputs[index]

    def calculate_error_by_target(self, target, activation_method, error_type):
        return self.calculate_error_derivative(target, error_type) * self.activation_derivative(activation_method)

    def calculate_error(self, target, error_type):
        if error_type.lower() == 'mse':
            return self.mse_error(target)
        else:
            raise Exception('Wrong error type')

    def calculate_error_derivative(self, target, error_type):
        if error_type.lower() == 'mse':
            return self.mse_error_derivative(target)
        else:
            raise Exception('Wrong error type')

    def activation(self, x, activation_method):
        if activation_method.lower() == 'sigmoid':
            return self.sigmoid(x)
        elif activation_method.lower() == 'relu':
            return self.relu(x)
        elif activation_method.lower() == 'linear':
            return self.linear(x)
        else:
            raise Exception('Wrong activation function specified')

    def activation_derivative(self, activation_method):
        if activation_method.lower() == 'sigmoid':
            return self.sigmoid_derivative()
        elif activation_method.lower() == 'relu':
            return self.relu_derivative()
        elif activation_method.lower() == 'linear':
            return self.linear_derivative()
        else:
            raise Exception('Wrong activation function specified')
