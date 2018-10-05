import random
import time
from agents import *

# global variables
board_size = 8
num_players = 4
num_food = 4
rounds = 100000
display = True
stop = True
speed = 0.4
# format = [row, col], starts at [0, 0]


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
    for r in range(1, rounds+1):
        if display:
            display_board(board)
            if stop:
                input()
            else:
                time.sleep(speed)
        play_round(board)
    if display:
        display_board(board)
        if stop:
            input()
        else:
            time.sleep(speed)
    print("All scores:", board.score)
    print("Team scores:", [board.score[0]+board.score[1], board.score[2]+board.score[3]])
    print(time.time() - t, "seconds")


if __name__ == "__main__":
    main()
