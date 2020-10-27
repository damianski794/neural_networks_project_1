import numpy as np
import matplotlib.pyplot as plt
from src.NeuralNet import NeuralNet
from src.DataReader import DataReader


class Classification(object):
    def __init__(self):
        self.data_reader = DataReader()

    def train(self, train_file, iterations, number_of_hidden_layers, number_of_hidden_nodes, if_bias, learning_rate, activation_method, error_type, img_name):
        inputs, outputs = self.data_reader.read_classification_data(train_file)
        inputs = np.array(inputs)
        outputs = np.array(outputs).T

        number_of_outputs = 0
        for i in range(len(outputs)):
            if outputs[i] > number_of_outputs:
                number_of_outputs = outputs[i]
        self.neural_net = NeuralNet(len(inputs[0]), number_of_hidden_layers, number_of_hidden_nodes, number_of_outputs, if_bias, learning_rate, activation_method, error_type, 'c')
        errors = self.neural_net.bulk_train(inputs, outputs, iterations)

        print(errors)
        plt.clf()
        plt.scatter(list(range(iterations)), errors, color='blue', s=10)
        plt.show()
        # plt.savefig('img/' + img_name)

    def test(self, test_file, img_name):
        inputs_test, outputs_test = self.data_reader.read_classification_data(test_file)
        inputs_test = np.array(inputs_test)
        outputs_test = np.array(outputs_test).T
        results = self.neural_net.bulk_test(inputs_test)
        x, y = map(list, zip(*inputs_test))
        x = list(map(float, x))
        y = list(map(float, y))
        colormap = np.array(['r', 'g', 'b', 'y', 'c', 'm'])
        for i in range(len(results)):
            if outputs_test[i] != results[i]:
                results[i] = 0
        plt.clf()
        plt.scatter(x, y, c=colormap[results])
        plt.show()
        # plt.savefig('img/' + img_name)
