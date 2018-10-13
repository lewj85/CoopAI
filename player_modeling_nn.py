import numpy as np
import json
from keras.models import Model, load_model
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.regularizers import l2
from keras.optimizers import SGD


board_size = 6
filename = 'data_6x6_1mil.json'


def main():
    # Create Player Model CNN
    print("making model")
    model = make_cnn()

    # Create computational graph
    print("creating graph")
    lr = 0.001
    sgd = SGD(lr=lr, decay=0.0, momentum=0.0, nesterov=False)
    model.compile(optimizer=sgd, loss='mse', metrics=['mse', 'accuracy'])

    # Load json
    print("loading json")
    with open(filename) as f:
        raw_data = json.load(f)

    # Prep data
    print("prepping data")
    X, y = prep_data(raw_data)

    # Train the model
    print("training model")
    epochs = 100
    batch_size = 1
    model.fit(x=X, y=y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=2)

    # Save each epoch in case we want to end early
    model.save('./cnn1_6x6_filt100_kern5_str1_wdec001_lr001_sgd_ep100.h5')


def make_cnn():
    # number of features for state info
    i1 = Input(shape=(board_size, board_size, 4), name='i1')
    x1 = cnn_part(i1)

    # 1st Hidden layer is another cnn
    # i2 = Input(shape=(board_size, board_size, 4), name='i2')
    # x2 = cnn_part(i2)

    # 2nd hidden layer is fully-connected
    num_hidden_nodes1 = 1000
    z1 = dense_part(x1, num_hidden_nodes1, 'relu')

    # Output layer
    output_final = dense_part(z1, 5, 'softmax')
    model = Model(inputs=i1, outputs=output_final)
    return model


def cnn_part(the_input):
    filters = 100
    kernel_size = 5
    stride = 1
    weight_decay = 0.001
    net = Conv2D(filters, kernel_size, strides=stride, kernel_initializer='random_uniform',
                 kernel_regularizer=l2(weight_decay))(the_input)
    net = Flatten()(net)  # don't need?
    return net


def dense_part(a, node_num, activation_type):
    net = Dense(units=node_num, activation=activation_type, kernel_initializer='random_uniform')(a)
    return net


def prep_data(raw_data):
    states = len(raw_data)
    X = np.zeros((states, board_size, board_size, 4))
    y = np.zeros((states, 5))
    for d in range(states):
        X[d] = np.asarray(raw_data[d]['board'])
        # one-hot encode the Ys for softmax outputs
        y[d][int(raw_data[d]['actions'][1])] = 1  # 2nd player is teammate
    return X, y


main()
