import random
import time
from agents import *

# global variables
BOARD_SIZE = 8
NUM_PLAYERS = 4
NUM_FOOD = 4
ROUNDS = 100000
DISPLAY = False
STOP = False
SPEED = 0.4
# format = [row, col], starts at [1, 1]


class State:
    def __init__(self):
        self.players = []
        self.foods = []
        for p in range(NUM_PLAYERS):
            self.players.append(GreedyClockwiseAgent(find_empty_spot(self)))
        for f in range(NUM_FOOD):
            self.foods.append(Food(find_empty_spot(self)))
        self.round = 1
        self.score = [0] * NUM_PLAYERS


class Food:
    def __init__(self, position):
        self.position = position


def find_empty_spot(board):
    while True:
        spot = [random.randint(1, BOARD_SIZE), random.randint(1, BOARD_SIZE)]
        if (spot not in [p.position for p in board.players]) and (spot not in [f.position for f in board.foods]):
            return spot


def play_round(board):
    # get new desired positions
    desired_spaces = [[0, 0]] * NUM_PLAYERS
    for i in range(NUM_PLAYERS):
        desired_spaces[i] = board.players[i].move(board)

    # move players, award points, and track eaten foods
    eaten = []
    for i in range(NUM_PLAYERS):

        # check for food
        ate_food = False
        for f in range(NUM_FOOD):
            if desired_spaces[i] == board.foods[f].position:
                ate_food = True
                if f not in eaten:
                    eaten.append(f)
                break

        # check for collision
        collided = False
        for j in range(NUM_PLAYERS):
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

    # if stuck for more than 1 turn, pick a different direction
    for i in range(NUM_PLAYERS):
        if board.players[i].stuck:
            board.players[i].position = find_empty_spot(board)
            board.players[i].stuck = False

    board.round += 1
    return board


def display_board(board):
    print("\n\n\n\n" + ". " * (BOARD_SIZE+1))
    for row in range(1, BOARD_SIZE+1):
        if row == BOARD_SIZE:
            row_string = "" + (". " * (BOARD_SIZE + 1))
        else:
            row_string = ". " + ("  " * (BOARD_SIZE - 1)) + "."
        i = 1
        for player in board.players:
            if player.position[0] == row:
                row_string = row_string[:player.position[1]*2-1] + str(i) + row_string[player.position[1]*2:]
            i += 1
        for food in board.foods:
            if food.position[0] == row:
                row_string = row_string[:food.position[1]*2-1] + "'" + row_string[food.position[1]*2:]
        print(row_string)
    print("round:", board.round)
    print("score:", board.score)


def main():
    # initialize the board
    board = State()

    # play some rounds
    t = time.time()
    for r in range(1, ROUNDS+1):
        if DISPLAY:
            display_board(board)
            if STOP:
                input()
            else:
                time.sleep(SPEED)
        board = play_round(board)

    if DISPLAY:
        display_board(board)
        if STOP:
            input()
        else:
            time.sleep(SPEED)
    print("All scores:", board.score)
    print("Team scores:", [board.score[0]+board.score[1], board.score[2]+board.score[3]])
    print(time.time() - t, "seconds")


if __name__ == "__main__":
    main()
