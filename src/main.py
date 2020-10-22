from src.Classification import Classification
from src.Regression import Regression


def klasyfikacja():
    TRAIN_FILE = 'data.simple.train.100.csv'
    TEST_FILE = 'data.simple.test.100.csv'
    ITERATIONS = 100
    NUMBER_OF_HIDDEN_LAYERS = 3
    NUMBER_OF_HIDDEN_NODES = 4
    IF_BIAS = True
    LEARNING_RATE = 0.01
    ACTIVATION_METHOD = 'linear'
    ERROR_TYPE = 'MSE'

    classification = Classification()
    classification.train(TRAIN_FILE, ITERATIONS, NUMBER_OF_HIDDEN_LAYERS, NUMBER_OF_HIDDEN_NODES, IF_BIAS, LEARNING_RATE, ACTIVATION_METHOD, ERROR_TYPE)
    classification.test(TEST_FILE)

def regresja():
    TRAIN_FILE = 'data.activation.train.100.csv'
    TEST_FILE = 'data.activation.test.100.csv'
    ITERATIONS = 1000
    NUMBER_OF_HIDDEN_LAYERS = 3
    NUMBER_OF_HIDDEN_NODES = 4
    IF_BIAS = True
    LEARNING_RATE = 0.0001
    ACTIVATION_METHOD = 'sigmoid'
    ERROR_TYPE = 'MSE'

    regression = Regression()
    regression.train(TRAIN_FILE, ITERATIONS, NUMBER_OF_HIDDEN_LAYERS, NUMBER_OF_HIDDEN_NODES, IF_BIAS,
                         LEARNING_RATE, ACTIVATION_METHOD, ERROR_TYPE)
    regression.test(TEST_FILE)

klasyfikacja()
#regresja()


