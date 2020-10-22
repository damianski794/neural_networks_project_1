import numpy as np
import matplotlib.pyplot as plt
from src.NeuralNet import NeuralNet
from src.DataReader import DataReader


class Regression(object):
    def __init__(self):
        self.data_reader = DataReader()

    def train(self, train_file, iterations, number_of_hidden_layers, number_of_hidden_nodes, if_bias, learning_rate, activation_method, error_type):
        inputs, outputs = self.data_reader.read_regression_data(train_file)
        inputs = np.array(inputs)
        outputs = np.array(outputs).T
        number_of_outputs = 1
        number_of_inputs = 1

        self.neural_net = NeuralNet(number_of_inputs, number_of_hidden_layers, number_of_hidden_nodes, number_of_outputs, if_bias, learning_rate, activation_method, error_type, 'r')
        errors = self.neural_net.bulk_train(inputs, outputs, iterations)

        print(errors)
        plt.scatter(list(range(iterations)), errors, color='blue', s=10)
        plt.show()

    def test(self, test_file):
        inputs_test, outputs_test = self.data_reader.read_regression_data(test_file)
        inputs_test = np.array(inputs_test)
        outputs_test = np.array(outputs_test).T
        results = self.neural_net.bulk_test(inputs_test)
        print(results)

        plt.scatter(inputs_test, outputs_test, color='blue', s=10)
        plt.scatter(inputs_test, results, color='red', s=0.5)
        plt.show()
