import csv
import numpy as np


class DataReader(object):
    def __init__(self):
        self.classificationPath = '../classification/'
        self.regressionPath = '../regression/'

    def read_regression_data(self, filename):
        with open(self.regressionPath + filename, newline='') as csvfile:
            data = list(csv.reader(csvfile))

        data = np.delete(data, 0, axis=0)
        x, y = map(list, zip(*data))

        inputs = zip(list(map(float, x)))
        outputs = list(map(float, y))

        return tuple(inputs), outputs

    def read_classification_data(self, filename):
        with open(self.classificationPath + filename, newline='') as csvfile:
            data = list(csv.reader(csvfile))

        data = np.delete(data, 0, axis=0)
        x, y, z = map(list, zip(*data))
        inputs = zip(list(map(float, x)), list(map(float, y)))
        outputs = list(map(int, z))

        return tuple(inputs), outputs
