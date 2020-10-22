import random as rand
from src.Neuron import Neuron


class NeuronLayer(object):
    def __init__(self, number_of_neurons, if_bias):
        self.bias = rand.random() if if_bias else 0
        self.neurons = []
        for i in range(number_of_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def calculate(self, inputs, activation_method):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate(inputs, activation_method))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.last_result)
        return outputs

    def init_weights(self, number_of_inputs):
        for neuron in self.neurons:
            for j in range(number_of_inputs):
                neuron.weights.append(rand.random())

    def update_output_values(self, results):
        for i in range(len(self.neurons)):
            self.neurons[i].last_result = results[i]
