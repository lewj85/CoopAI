import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Dense, Flatten, Multiply
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from game import State, Food, find_empty_spot, play_round
import json
import random

# globals
board_size = 8
n_actions = 5
# loadpath = "dqn8_f32k5_f32k3_rms_lr00025.100-02.hdf5"
savepath = "dqn8_f32k5_f32k3_rms_lr00025.{epoch:02d}-{val_loss:.2f}.hdf5"
memorypath = "8cnn.json"
batch_size = 32
gamma = 0.99 # discount factor


def main():
    print("8dqn")

    # Create DQN
    print("making model")
    model = make_dqn()
    # model = load_model(loadpath)

    # Create computational graph
    print("creating graph")
    lr = 0.00025
    # sgd = SGD(lr=lr, decay=0.0, momentum=0.0, nesterov=False)
    # model.compile(optimizer=sgd, loss='mse', metrics=['mse', 'accuracy'])
    rms = RMSprop(lr=lr, rho=0.95, epsilon=0.01)
    model.compile(optimizer=rms, loss='mse', metrics=['mse', 'accuracy'])

    # Load memory and starting state
    # memory = json.loads(memorypath)
    memory = []
    state = State()

    # Train model
    iterations = 1000
    for iteration in range(iterations):
        q_iteration(model, memory, iteration)


def make_dqn():
    all_combos = n_actions * n_actions
    # Inputs - the 4 is one-hot encoded [player, ally, enemy, food]
    frames_input = Input((board_size, board_size, 4), name='frames')
    actions_input = Input((all_combos,), name='mask')
    # 1st hidden layer is a large cnn
    conv1 = Conv2D(32, 5, strides=1, kernel_initializer='random_uniform')(frames_input)
    # 2nd hidden layer is a more local cnn
    conv2 = Conv2D(32, 3, strides=1, kernel_initializer='random_uniform')(conv1)
    # flatten after convolutions so all dense layers have 2 dimensions
    conv_flattened = Flatten()(conv2)
    # 3rd hidden layer is fully-connected
    hidden = Dense(128, activation='relu')(conv_flattened)
    # Output layer
    output = Dense(all_combos)(hidden)
    # Multiply the output by the mask
    filtered_output = Multiply()([output, actions_input])

    model = Model(input=[frames_input, actions_input], output=filtered_output)
    return model


def q_iteration(model, state, memory, iteration):
    # Get random batch
    if len(memory) > batch_size:
        start_states = random.sample(memory, batch_size)

    # epsilon goes from 1 down to 0.1 over 900,000 rounds
    if iteration > 900000:
        epsilon = 0.1
    else:
        epsilon = (1000000 - iteration) / 1000000

    # Choose the action: random action if < epsilon or best action if > epsilon
    if random.random() < epsilon:
        action = random.randint(0, 4)
    else:
        q_vals = model.predict([next_states, np.ones(actions.shape)])
        action = q_vals.index(max(q_vals))

    # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
    new_frame, reward, is_done, _ = env.step(action)
    memory.add(state, action, new_)

    # Fit (when there are enough samples)
    if len(memory) > batch_size:
        fit_batch(model, start_states, actions, rewards, next_states)


def fit_batch(model, start_states, actions, rewards, next_states):
    """Do one deep Q learning iteration
    Params:
    - model: The DQN
    - start_states: numpy array of starting states
    - actions: numpy array of one-hot encoded actions corresponding to the start states
    - rewards: numpy array of rewards corresponding to the start states and actions
    - next_states: numpy array of the resulting states corresponding to the start states and actions
    - is_terminal: numpy boolean array of whether the resulting state is terminal
    """
    next_Q_values = model.predict([next_states, np.ones(actions.shape)])
    # The Q values of each start state is the reward + gamma * the max next state Q value
    Q_values = rewards + gamma * np.max(next_Q_values, axis=1)
    # Fit the keras model. Note how we are passing the actions as the mask and multiplying
    # the targets by the actions.
    model.fit([start_states, actions], actions * Q_values[:, None], epochs=1, batch_size=len(start_states), verbose=2)

    # # Create checkpoints
    # checkpoint = ModelCheckpoint(savepath, monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=False,
    #                              mode='auto', period=1)
    # callbacks_list = [checkpoint]
    #
    # # Train the model
    # print("training model")
    # model.fit(x=X, y=y, epochs=1, batch_size=1, validation_split=0.2, verbose=2, callbacks=callbacks_list)
