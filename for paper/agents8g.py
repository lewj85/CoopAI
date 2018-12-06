from game8g import board_size, play_round, num_food, get_desired_space_from_action
import numpy as np
import random
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Dense, Flatten, Multiply
from keras.optimizers import SGD, RMSprop, Nadam
from keras import backend as K
from collections.abc import Sequence


action_dict = {
    0: "up",
    1: "right",
    2: "down",
    3: "left",
    4: "stay"
}


class Memory(Sequence):
    def __init__(self, size):
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY, whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. This greatly simplifies the rest of the code.
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0

    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]

    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class GreedyClockwiseAgent:
    def __init__(self, position):
        self.position = position
        self.target = None
        self.action = None
        self.stuck = False

    def pick_target(self, board):
        self.target = [None, board_size * 2]
        for food in board.foods:
            # manhattan distance
            # NOTE: does not take into account obstacles (other players)
            distance = abs(self.position[0] - food.position[0]) + abs(self.position[1] - food.position[1])
            if distance < self.target[1]:
                self.target = [food, distance]
        return self.target

    def move(self, board):
        desired_position = self.position
        # find closest food
        self.pick_target(board)
        # move toward it in clockwise preference: up > right > down > left
        moved = False
        # check up
        if self.position[0] > self.target[0].position[0]:
            # make sure you stay on the board
            if self.position[0]-1 >= 0:
                desired_position = [self.position[0]-1, self.position[1]]
                self.action = 0
                moved = True
        # check right
        if not moved and self.position[1] < self.target[0].position[1]:
            # make sure you stay on the board
            if self.position[1]+1 <= board_size-1:
                desired_position = [self.position[0], self.position[1]+1]
                self.action = 1
                moved = True
        # check down
        if not moved and self.position[0] < self.target[0].position[0]:
            # make sure you stay on the board
            if self.position[0]+1 <= board_size-1:
                desired_position = [self.position[0]+1, self.position[1]]
                self.action = 2
                moved = True
        # check left
        if not moved and self.position[1] > self.target[0].position[1]:
            # make sure you stay on the board
            if self.position[1]-1 >= 0:
                desired_position = [self.position[0], self.position[1]-1]
                self.action = 3
                moved = True
        # if stuck against a wall and can't move, stand still (ie. don't change desired_position)
        if not moved:
            self.action = 4
        return desired_position


class CooperativeAI:
    def __init__(self, position):
        self.position = position
        # self.target = None
        self.action = None
        self.stuck = False
        self.num_actions = 5
        # self.loadpath_cnn = "models/cnn8_f32k5_f32k3_lr0001.99-0.02.hdf5"
        # self.loadpath_dqn = "models/dqn8_f32k5_f32k3_lr00025 - 900000 - 0.1960301250219345.hdf5"
        self.loadpath_dqn = "models/dqn8d_f128k5_sgdn_lr00001_rewardself10 - 2000000 - WORKS.hdf5"
        self.savepath = "dqn8g_f256k5_sgdn_lr00001_rewardall10_batch8"
        # self.model_cnn = load_model(self.loadpath_cnn)
        self.model_dqn = self.make_dqn()
        # self.model_dqn = load_model(self.loadpath_dqn)
        lr = 0.00001
        optimizer = SGD(lr=lr, decay=0.0, momentum=0.0, nesterov=True)
        # optimizer = RMSprop(lr=lr, rho=0.95, epsilon=0.01)
        # optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        self.model_dqn.compile(optimizer=optimizer, loss='mse')
        # don't compile after loading
        # self.model_dqn.optimizer = optimizer
        # K.set_value(self.model_dqn.optimizer.lr, lr)
        print("8dqn, sgdn, 0.00001, 256k5, 128 hidden, batch 8, rewardall10")
        self.memory = Memory(500000)
        self.min_memory_size = 50000
        self.batch_size = 8
        self.iteration = 0
        self.epsilon = 1
        self.gamma = 0.8
        self.mask1 = np.ones((1, 25))
        self.mask16 = np.ones((self.batch_size, 25))
        self.best_loss = 10000

    def make_dqn(self):
        all_combos = self.num_actions * self.num_actions
        # Inputs - the 4 is one-hot encoded [player, ally, enemy, food]
        frames_input = Input((board_size, board_size, 4), name='frames')
        actions_input = Input((all_combos,), name='mask')
        # 1st hidden layer is a large cnn
        conv1 = Conv2D(256, 5, strides=1, kernel_initializer='random_uniform')(frames_input)
        # 2nd hidden layer is a more local cnn
        # conv2 = Conv2D(32, 3, strides=1, kernel_initializer='random_uniform')(conv1)
        # flatten after convolutions so all dense layers have 2 dimensions
        conv_flattened = Flatten()(conv1)
        # 3rd hidden layer is fully-connected
        hidden = Dense(128, activation='relu')(conv_flattened)
        # Output layer
        output = Dense(all_combos)(hidden)
        # Multiply the output by the mask
        filtered_output = Multiply()([output, actions_input])
        model = Model(input=[frames_input, actions_input], output=filtered_output)
        return model

    def make_np_array(self, board):
        # the 4 is one-hot encoded [player, ally, enemy, food]
        np_board = np.zeros((1, board_size, board_size, 4), dtype=np.uint8)
        # players
        np_board[0][board.players[0].position[0]][board.players[0].position[1]][1] = 1
        np_board[0][board.players[1].position[0]][board.players[1].position[1]][0] = 1
        np_board[0][board.players[2].position[0]][board.players[2].position[1]][2] = 1
        np_board[0][board.players[3].position[0]][board.players[3].position[1]][2] = 1
        # food
        for i in range(num_food):
            np_board[0][board.foods[i].position[0]][board.foods[i].position[1]][3] = 1
        return np_board

    def move(self, board):

        # Convert to numpy array
        start_state = self.make_np_array(board)

        # epsilon goes from 1 down to 0.1 over 900,000 rounds
        if self.epsilon > 0.1:
            self.epsilon -= 0.000001

        # get Q values
        all_q_vals = self.model_dqn.predict([start_state, self.mask1])

        # get ally's predicted move
        # ally_predicted_action = self.model_cnn.predict(start_state)
        # print("ally", ally_predicted_action)
        # identify which 5 output nodes to take max from
        # ally_alist = ally_predicted_action[0].tolist()
        # ally_action = ally_alist.index(max(ally_alist))
        # start_index = ally_action * self.num_actions

        # TODO Remove later, add cnn back
        # get ally's next move
        tmp_board = play_round(board, 4)
        ally_action = tmp_board.players[0].action
        start_index = ally_action * self.num_actions

        # Choose your action: random action if < epsilon or best action if > epsilon
        rand_val = random.random()
        if rand_val < self.epsilon:
            action = random.randint(0, 4)
        else:
            if ally_action == self.num_actions - 1:
                q_vals = all_q_vals[0][start_index:]
            else:
                q_vals = all_q_vals[0][start_index:start_index + self.num_actions]
            qlist = q_vals.tolist()
            action = qlist.index(max(qlist))

        # Play one game iteration with the chosen action
        new_board = play_round(board, action)
        new_state = self.make_np_array(new_board)

        #######################
        #  reward calculation
        #######################
        reward = 0
        # first add actual points (use team scores rather than just the player's score)
        # team 1 score minus team 2 score
        reward += (new_board.score[0] - board.score[0] + new_board.score[1] - board.score[1] -
                  (new_board.score[2] - board.score[2] + new_board.score[3] - board.score[3])) * 10
        # reward += (new_board.score[0] - board.score[0] + new_board.score[1] - board.score[1]) * 10

        # get everyone else's targets
        # find your target based on closest food in direction of action taken (don't consider foods 'behind' you)
        # consider that you could have just eaten your target and it moved
        # if you and ally share same target, lose points, unless any enemy is closer to the same target than your ally
        # very small reward for moving toward center of map
        # other rewards

        action_mask = np.zeros((1, 25), dtype=np.uint8)
        action_mask[0][start_index+action] = 1
        self.memory.append([start_state, action_mask, reward, new_state])

        # Fit (when there are enough samples)
        if len(self.memory) > self.min_memory_size:
            batch = random.sample(self.memory, self.batch_size)
            start_states = np.array([np.squeeze(x[0]) for x in batch], dtype=np.uint8) # remove first dimension
            actions =      np.array([np.squeeze(x[1]) for x in batch], dtype=np.uint8) # remove first dimension
            rewards =      np.array([x[2] for x in batch])
            next_states =  np.array([np.squeeze(x[3]) for x in batch], dtype=np.uint8) # remove first dimension
            self.fit_batch(start_states, actions, rewards, next_states)
            self.iteration += 1

        # Make a new prediction now that the weights have been updated
        all_q_vals = self.model_dqn.predict([start_state, self.mask1])
        if ally_action == self.num_actions - 1:
            q_vals = all_q_vals[0][start_index:]
        else:
            q_vals = all_q_vals[0][start_index:start_index + self.num_actions]
        qlist = q_vals.tolist()
        if self.iteration%10000==1:
            print(qlist)
        action = qlist.index(max(qlist))
        # print(action)
        return get_desired_space_from_action(self.position, action)

    def fit_batch(self, start_states, actions, rewards, next_states):
        """Do one deep Q learning iteration.
        Params:
        - start_states: numpy array of starting states
        - actions: numpy array of one-hot encoded actions corresponding to the start states
        - rewards: numpy array of rewards corresponding to the start states and actions
        - next_states: numpy array of the resulting states corresponding to the start states and actions
        """
        # First, predict the Q values of the next states. Note how we are passing ones as the mask.
        next_Q_values = self.model_dqn.predict([next_states, self.mask16])
        # The Q values of each start state is the reward + gamma * the max next state Q value
        Q_values = rewards + self.gamma * np.max(next_Q_values, axis=1)
        # Fit the keras model. Note how we are passing the actions as the mask and multiplying
        # the targets by the actions.
        X = [start_states, actions]
        y = actions * Q_values[:, None]
        self.model_dqn.fit(X, y, epochs=1, batch_size=len(start_states), verbose=0)
        # print feedback every 100000 iterations
        if self.iteration % 100000 == 0:
            new_loss = self.model_dqn.evaluate(X, y, batch_size=len(start_states), verbose=0)
            print(new_loss, "b")
            self.save_dqn(new_loss)

    def save_dqn(self, new_loss):
        path = self.savepath + " - " + str(self.iteration) + " - " + str(new_loss) + ".hdf5"
        self.model_dqn.save(path)
