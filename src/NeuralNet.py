import numpy as np
from src.NeuronLayer import NeuronLayer


class NeuralNet(object):
    def __init__(self, number_of_inputs, number_of_hidden_layers, number_of_hidden_nodes, number_of_outputs, if_bias, learning_rate, activation_method, error_type, type):
        self.number_of_inputs = number_of_inputs
        self.learning_rate = learning_rate
        self.number_of_outputs = number_of_outputs
        self.number_of_hidden_layers = number_of_hidden_layers
        self.hidden_layers = []
        self.activation_method = activation_method
        self.error_type = error_type
        self.type = type

        for i in range(number_of_hidden_layers):
            self.hidden_layers.append(NeuronLayer(number_of_hidden_nodes, if_bias))

        self.output_layer = NeuronLayer(number_of_outputs, if_bias)

        if number_of_hidden_layers > 0:
            self.hidden_layers[0].init_weights(number_of_inputs)
            for i in range(1, number_of_hidden_layers):
                self.hidden_layers[i].init_weights(number_of_hidden_nodes)
            self.hidden_layers = np.array(self.hidden_layers)
            self.output_layer.init_weights(number_of_hidden_nodes)
        else:
            self.output_layer.init_weights(number_of_inputs)

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.number_of_inputs))
        print('------')
        for i in range(self.number_of_hidden_layers):
            print('Hidden Layer')
            self.hidden_layers[i].inspect()
            print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def softmax(self, x):
        expo = np.exp(x)
        expo_sum = np.sum(np.exp(x))
        return expo / expo_sum

    def calculate(self, inputs):
        if self.number_of_hidden_layers > 0:
            hidden_layer_outputs = self.hidden_layers[0].calculate(inputs, self.activation_method)
            for i in range(1, self.number_of_hidden_layers):
                hidden_layer_outputs = self.hidden_layers[i].calculate(hidden_layer_outputs, self.activation_method)
            if self.type == 'c':
                results = self.softmax(self.output_layer.calculate(hidden_layer_outputs, 'linear'))
                self.output_layer.update_output_values(results)
            elif self.type == 'r':
                results = self.output_layer.calculate(hidden_layer_outputs, 'linear')
            else:
                raise Exception('Wrong type')
            return results
        else:
            if self.type == 'c':
                results = self.softmax(self.output_layer.calculate(inputs, 'linear'))
                self.output_layer.update_output_values(results)
            elif self.type == 'r':
                results = self.output_layer.calculate(inputs, 'linear')
            else:
                raise Exception('Wrong type')
            return results

    def return_answer(self, inputs):
        result = self.calculate(inputs)
        if self.type == 'r':
            return result
        answer = -1
        probability = -1
        for i in range(len(result)):
            if result[i] > probability:
                answer = i
                probability = result[i]
        return answer + 1

    def transform_outputs(self, output):
        if self.type == 'c':
            transformed_output = np.zeros(self.number_of_outputs)
            transformed_output[output-1] = 1
            return transformed_output
        if self.type == 'r':
            return [output]

    def train(self, training_inputs, training_outputs):
        self.calculate(training_inputs)
        training_outputs = self.transform_outputs(training_outputs)

        # Output neuron deltas
        output_deltas = np.zeros(len(self.output_layer.neurons))
        for i in range(len(self.output_layer.neurons)):
            output_deltas[i] = self.output_layer.neurons[i].calculate_error_by_target(training_outputs[i], self.activation_method, self.error_type)

        # Hidden neuron deltas
        hidden_deltas = []
        for k in range(self.number_of_hidden_layers):
            hidden_delta = np.zeros(len(self.hidden_layers[k].neurons))
            for i in range(len(self.hidden_layers[k].neurons)):
                derivative_of_neuron = 0
                for j in range(len(self.output_layer.neurons)):
                    derivative_of_neuron += output_deltas[j] * self.output_layer.neurons[j].weights[i]
                hidden_delta[i] = derivative_of_neuron * self.hidden_layers[k].neurons[i].activation_derivative(self.activation_method)
            hidden_deltas.append(hidden_delta)

        # Update output neuron weights
        for i in range(len(self.output_layer.neurons)):
            for j in range(len(self.output_layer.neurons[i].weights)):
                error = output_deltas[i] * self.output_layer.neurons[i].get_last_input_by_index(j)
                self.output_layer.neurons[i].weights[j] -= self.learning_rate * error

        # Update hidden neuron weights
        for k in range(self.number_of_hidden_layers):
            for i in range(len(self.hidden_layers[k].neurons)):
                for j in range(len(self.hidden_layers[k].neurons[i].weights)):
                    error = hidden_deltas[k][i] * self.hidden_layers[k].neurons[i].get_last_input_by_index(j)
                    self.hidden_layers[k].neurons[i].weights[j] -= self.learning_rate * error

        return self.calculate_total_error(training_outputs)

    def calculate_total_error(self, target_output):
        total_error = 0
        for i in range(len(self.output_layer.neurons)):
            total_error += self.output_layer.neurons[i].calculate_error(target_output[i], self.error_type)
        return total_error

    def bulk_train(self, inputs, outputs, iterations):
        errors = []
        for i in range(iterations):
            error = 0
            for j in range(len(inputs)):
                error += self.train(inputs[j], outputs[j])
            errors.append(error)
        return errors

    def bulk_test(self, inputs):
        answers = []
        for i in range(len(inputs)):
            answer = self.return_answer(inputs[i])
            answers.append(answer)
        return answers
