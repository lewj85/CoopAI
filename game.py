import random
import time
from agents import *
import numpy as np
import json

# global variables
board_size = 6
num_players = 4
num_food = 4
rounds = 1000
display = False
save_data = False
# format = [row, col], starts at [0, 0]
# players 1 + 2 vs 3 + 4


class State:
    def __init__(self):
        self.players = []
        self.foods = []
        for p in range(num_players):
            self.players.append(GreedyClockwiseAgent(find_empty_spot(self)))
        for f in range(num_food):
            self.foods.append(Food(find_empty_spot(self)))
        self.round = 1
        self.score = [0] * num_players


class Food:
    def __init__(self, position):
        self.position = position


def find_empty_spot(board):
    players = [p.position for p in board.players]
    foods = [f.position for f in board.foods]
    while True:
        spot = [random.randint(0, board_size-1), random.randint(0, board_size-1)]
        if (spot not in players) and (spot not in foods):
            return spot


def play_round(board):
    # get new desired positions
    desired_spaces = [[0, 0]] * num_players
    for i in range(num_players):
        desired_spaces[i] = board.players[i].move(board)
    # move players, award points, and track eaten foods
    eaten = []
    for i in range(num_players):
        # check for food
        ate_food = False
        for f in range(num_food):
            if desired_spaces[i] == board.foods[f].position:
                ate_food = True
                if f not in eaten:
                    eaten.append(f)
                break
        # check for collision
        collided = False
        for j in range(num_players):
            if desired_spaces[i] == desired_spaces[j]:
                if i != j:
                    collided = True
                    break
        # award points
        if ate_food:
            board.score[i] += 2
            if collided:
                board.score[i] -= 1
        # move the player
        if not collided:
            board.players[i].position = desired_spaces[i]
        else:
            board.players[i].stuck = True
    # move eaten foods
    for f in eaten:
        board.foods[f].position = find_empty_spot(board)
    # if stuck, move to a random empty spot
    for i in range(num_players):
        if board.players[i].stuck:
            board.players[i].position = find_empty_spot(board)
            board.players[i].stuck = False
    board.round += 1


def display_board(board):
    print(". " * (board_size+1))
    for row in range(0, board_size):
        if row == board_size-1:
            row_string = "" + (". " * (board_size + 1))
        else:
            row_string = ". " + ("  " * (board_size - 1)) + "."
        i = 1
        for player in board.players:
            if player.position[0] == row:
                row_string = row_string[:player.position[1]*2+1] + str(i) + row_string[player.position[1]*2+2:]
            i += 1
        for food in board.foods:
            if food.position[0] == row:
                row_string = row_string[:food.position[1]*2+1] + "'" + row_string[food.position[1]*2+2:]
        print(row_string)
    print("round:", board.round)
    print("score:", board.score)


def main():
    # initialize the board
    board = State()
    # play some rounds
    t = time.time()
    if save_data:
        f = open("data.json", "w+")
        f.write("[")
    for r in range(0, rounds):
        if save_data:
            positions = [p.position for p in board.players]
            targets = [[f[0].position, f[1]] for f in [p.pick_target(board) for p in board.players]]
            old_scores = [s for s in board.score]
            board_mtx = np.zeros([board_size, board_size, 4])
            # one-hot encoding: [player, ally, enemy, food]
            for i in range(len(board.players)):
                if i == 0:  # player
                    board_mtx[board.players[i].position[0]][board.players[i].position[1]][0] = 1
                elif i == 1:  # ally
                    board_mtx[board.players[i].position[0]][board.players[i].position[1]][1] = 1
                else:  # enemy
                    board_mtx[board.players[i].position[0]][board.players[i].position[1]][2] = 1
            for i in range(len(board.foods)):
                board_mtx[board.foods[i].position[0]][board.foods[i].position[1]][3] = 1
        if display:
            display_board(board)
            input()
        play_round(board)
        if save_data:
            actions = [p.action for p in board.players]
            new_scores = [s for s in board.score]
            data_dict = {'board': board_mtx.tolist(),
                         'positions': positions,
                         'targets': targets,
                         'old_scores': old_scores,
                         'actions': actions,
                         'new_scores': new_scores}
            data = json.dumps(data_dict)
            f.write(data)
            if r != rounds-1:
                f.write(", ")
            else:
                f.write("]")
                f.close()
    if display:
        display_board(board)
        input()
    print("All scores:", board.score)
    print("Team scores:", [board.score[0]+board.score[1], board.score[2]+board.score[3]])
    print(time.time() - t, "seconds")


if __name__ == "__main__":
    main()
