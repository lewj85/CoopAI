from game import board_size, play_round, num_food, get_desired_space_from_action
import numpy as np
from keras.models import load_model
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
        self.loadpath_dqn = "for paper/dqn8d3_f256k5_sgdn_lr00001_rewardself10 - 9800000 - 0.2657600939273834.hdf5"
        self.model_dqn = load_model(self.loadpath_dqn)
        self.mask1 = np.ones((1, 25))

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

        # Make a new prediction now that the weights have been updated
        all_q_vals = self.model_dqn.predict([start_state, self.mask1])
        if ally_action == self.num_actions - 1:
            q_vals = all_q_vals[0][start_index:]
        else:
            q_vals = all_q_vals[0][start_index:start_index + self.num_actions]
        qlist = q_vals.tolist()
        action = qlist.index(max(qlist))
        self.action = action
        return get_desired_space_from_action(self.position, action)
