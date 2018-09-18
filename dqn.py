from keras.models import Model, load_model
from keras.initializers import Constant
from keras.layers import Conv2D, Dense, Flatten, Input, concatenate
from keras.regularizers import l2
from keras.optimizers import Adam
import numpy as np
import json
import copy


def main():
    # Create DQN
    model = make_dqn()

    # Create computational graph
    lr = 0.0001
    sgd = optimizers.SGD(lr=lr, decay=0.0, momentum=0.0, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    # Prep data
    filename = 'game_data1.json'
    games = json.load(open(filename))
    num_games = len(games)

    epochs = 2
    for e in range(epochs):
        for g in range(num_games):
            # Prep the data
            data, win_or_loss = prep_data(games[g])

            # Train the model
            train_model(model, data, win_or_loss)

        # Save each epoch in case we want to end early
        model.save('./dqn1_331_0001_sgd_1000_a95d.h5')


def make_dqn():
    # number of features for state info = 13*13*12
    a1 = Input(shape=(13, 13, 12), name='a1')
    output1 = cnn_part(a1)
    # number of features for bomb timer info = 13*13
    a2 = Input(shape=(13, 13, 1), name='a2')
    output2 = cnn_part(a2)
    # number of features for flame timer info = 13*13
    a3 = Input(shape=(13, 13, 1), name='a3')
    output3 = cnn_part(a3)
    # number of other features = 8 + action_pair 12
    a4 = Input(shape=(20,), name='a4')
    alist = [output1, output2, output3, a4]
    # Join the inputs
    a5 = concatenate(alist, axis=-1)
    # Hidden layer
    all_as = dense_part(a5, 1000, 'relu')
    # Output layer
    output_final = dense_part(all_as, 1, 'linear')
    model = Model(inputs=[a1, a2, a3, a4], outputs=output_final)
    return model


def cnn_part(a):
    filters = 25
    kernel_size = 3
    stride = 1
    weight_decay = 0.01
    net = Conv2D(filters, kernel_size, strides=stride, kernel_initializer='random_uniform',
                 kernel_regularizer=l2(weight_decay))(a)
    net = Flatten()(net)
    return net


def dense_part(a, node_num, activation_type):
    net = Dense(units=node_num, activation=activation_type, kernel_initializer='random_uniform')(a)
    return net


def prep_data(game):
    states = len(game['states_w_action_pairs'])
    # print(states)
    features = len(game['states_w_action_pairs'][0])
    data = np.zeros((states, features))
    for d in range(states):
        for e in range(features):
            data[d][e] = game['states_w_action_pairs'][d][e]
    win_or_loss = int(game['reward'])
    shaped_data1 = np.reshape(data[:, :2028], (states, 13, 13, 12))
    shaped_data2 = np.reshape(data[:, 2028:2197], (states, 13, 13, 1))
    shaped_data3 = np.reshape(data[:, 2197:2366], (states, 13, 13, 1))
    shaped_data4 = np.reshape(data[:, 2366:], (states, 20))  # 2386
    data_cat = {'a1': shaped_data1, 'a2': shaped_data2, 'a3': shaped_data3, 'a4': shaped_data4}
    return data_cat, win_or_loss


def train_model(model, data, win_or_loss, batch_size=1, epochs=1):
    num_examples = data['a4'].shape[0]
    print(num_examples)

    all_a1 = data['a1']
    all_a2 = data['a2']
    all_a3 = data['a3']
    all_a4 = data['a4']

    r = 0
    diminishing_reward_value = 0.99
    alpha_learning_rate = 0.95
    decay = diminishing_reward_value ** np.arange(num_examples)

    # Train on one state at a time, starting from the end
    for i in range(num_examples - 1, -1, -1):

        X_train = {'a1': np.reshape(all_a1[i], (1, 13, 13, 12)),
                   'a2': np.reshape(all_a2[i], (1, 13, 13, 1)),
                   'a3': np.reshape(all_a3[i], (1, 13, 13, 1)),
                   'a4': np.reshape(all_a4[i], (1, 20))}
        y_train = np.zeros((1,))
        if i == num_examples - 1:
            y_train[0] = 2 * win_or_loss
            # q_prime = y_train[0]
        else:
            # find max q_prime
            q_primes = []
            X_prime = {}
            X_prime['a1'] = np.reshape(all_a1[i + 1], (1, 13, 13, 12))
            X_prime['a2'] = np.reshape(all_a2[i + 1], (1, 13, 13, 1))
            X_prime['a3'] = np.reshape(all_a3[i + 1], (1, 13, 13, 1))
            X_prime['a4'] = np.reshape(all_a4[i + 1], (1, 20))
            # first reset action taken
            for j in range(6):
                X_prime['a4'][0][-6 + j] = 0
            # get Q values for all 6 possible actions
            last_j = 0
            for j in range(6):
                X_prime['a4'][0][-6 + last_j] = 0
                X_prime['a4'][0][-6 + j] = 1
                q_primes.append(model.predict(X_prime))
                last_j = j
            q_prime = max(q_primes)

            # Q function
            y_train[0] = (1 - alpha_learning_rate) * model.predict(X_train) + \
                         alpha_learning_rate * (r + decay[i] * q_prime)

            # Training
            model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=batch_size, verbose=2)
