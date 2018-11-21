import numpy as np
import json
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint


board_size = 10
filename = '10cnn.json'
loadpath = "cnn10_f32k5_f32k3_lr001.100-0.02.hdf5"
filepath = "cnn10_f32k5_f32k3_lr0001.{epoch:02d}-{val_loss:.2f}.hdf5"


def main():

    print("8cnn")

    # Create Player Model CNN
    print("making model")
    # model = make_cnn()
    model = load_model(loadpath)

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

    # Create checkpoints
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    callbacks_list = [checkpoint]

    # Train the model
    print("training model")
    model.fit(x=X, y=y, epochs=100, batch_size=1, validation_split=0.2, verbose=2, callbacks=callbacks_list)


def make_cnn():
    model = Sequential()

    # Input layer is a cnn
    model.add(Conv2D(32, 5, strides=1, input_shape=(board_size, board_size, 4), kernel_initializer='random_uniform'))
    
    # 1st hidden layer is a more local cnn
    model.add(Conv2D(32, 3, strides=1, kernel_initializer='random_uniform'))

    # flatten after convolutions so all dense layers have 2 dimensions
    model.add(Flatten())

    # 2nd hidden layer is fully-connected
    model.add(Dense(units=100, activation='relu', kernel_initializer='random_uniform'))

    # Output layer
    model.add(Dense(units=5, activation='softmax', kernel_initializer='random_uniform'))
    return model


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
