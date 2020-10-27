import time
from src.Classification import Classification


def klasyfikacja():
    TRAIN_FILE = 'data.XOR.train.100.csv'
    TEST_FILE = 'data.XOR.test.100.csv'
    NUMBER_OF_HIDDEN_LAYERS = 0
    NUMBER_OF_HIDDEN_NODES = 10
    ACTIVATION_METHOD = 'relu'
    ERROR_TYPE = 'MSE'

    ITERATIONS = 1000
    LEARNING_RATE = 0.001
    IF_BIAS = True

    start_time = time.time()
    classification = Classification()
    classification.train(TRAIN_FILE, ITERATIONS, NUMBER_OF_HIDDEN_LAYERS, NUMBER_OF_HIDDEN_NODES, IF_BIAS, LEARNING_RATE, ACTIVATION_METHOD, ERROR_TYPE, 'cat')
    classification.test(TEST_FILE, 'dog')
    print("--- %s seconds ---" % (time.time() - start_time))




def training():
    activation_array = ['relu']
    train_array = [['data.three_gauss.train.10000.csv', 'data.three_gauss.test.10000.csv', 'gauss10000']]

    neurons = [4]
    layers = [2, 4]

    for train in train_array:
        for act in activation_array:
            for lay in layers:
                for nod in neurons:
                    if lay == 0 and nod == 2:
                        classification = Classification()
                        img_name_errors = '3' + train[2] + act + '_layers' + str(
                            lay) + '_nodes0' + '_mae' + '_errors' + '.png'
                        img_name_class = '3' + train[2] + act + '_layers' + str(
                            lay) + '_nodes0' + '_mae' + '_classification' + '.png'
                        classification.train(train[0], 5000, lay, nod, True, 0.001, act, 'mae', img_name_errors)
                        classification.test(train[1], img_name_class)
                    elif lay == 0:
                        continue
                    else:
                        classification = Classification()
                        img_name_errors = '5' + train[2] + act + '_layers' + str(lay) + '_nodes' + str(
                            nod) + '_mse' + '_errors' + '.png'
                        img_name_class = '5' + train[2] + act + '_layers' + str(lay) + '_nodes' + str(
                            nod) + '_mse' + '_classification' + '.png'
                        classification.train(train[0], 5000, lay, nod, True, 0.001, act, 'mse', img_name_errors)
                        classification.test(train[1], img_name_class)


klasyfikacja()
# training()


