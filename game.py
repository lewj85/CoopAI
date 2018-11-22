import random
# import time
import numpy as np
import json

# consider adding a punishment to 'require' cooperation, such as losing 5 points if you and your teammate eat food on the same round
# consider adding multiplication of ally_predicted_action probability and subtracting the worst q_val from each 'grouping' of 5

# global variables
board_size = 8
num_players = 4
num_food = 4
rounds = 1100001
display = False
save_data = False
# format = [row, col], NOT [x, y]! starts at [0, 0]
# players 1 + 2 vs 3 + 4


def get_desired_space_from_action(old_pos, action):
    #0: "up",
    #1: "right",
    #2: "down",
    #3: "left",
    #4: "stay"
    # [row, col], NOT [x, y]
    if action == 0:
        new_pos = [old_pos[0] - 1, old_pos[1]]
    elif action == 1:
        new_pos = [old_pos[0], old_pos[1] + 1]
    elif action == 2:
        new_pos = [old_pos[0] + 1, old_pos[1]]
    elif action == 3:
        new_pos = [old_pos[0], old_pos[1] - 1]
    elif action == 4:
        new_pos = old_pos
    return new_pos


class State:
    def __init__(self, copy_players=None, copy_foods=None, copy_rounds=None, copy_score=None):
        self.players = []
        self.foods = []
        from agents5 import GreedyClockwiseAgent, CooperativeAI
        for p in range(num_players):
            if copy_players:
                self.players.append(GreedyClockwiseAgent(copy_players[p]))
            else:
                if p == 1:
                    self.players.append(CooperativeAI(find_empty_spot(self)))
                else:
                    self.players.append(GreedyClockwiseAgent(find_empty_spot(self)))
        for f in range(num_food):
            if copy_foods:
                self.foods.append(Food(copy_foods[f]))
            else:
                self.foods.append(Food(find_empty_spot(self)))
        if copy_rounds:
            self.round = copy_rounds
        else:
            self.round = 1
        if copy_score:
            self.score = copy_score
        else:
            self.score = [0] * num_players


class Food:
    def __init__(self, position):
        self.position = position


# returns [row, col] position, not [x, y]
def find_empty_spot(board):
    players = [p.position for p in board.players]
    foods = [f.position for f in board.foods]
    while True:
        spot = [random.randint(0, board_size-1), random.randint(0, board_size-1)]
        if (spot not in players) and (spot not in foods):
            return spot


def play_round(original_board, action=-1):
    # in case we're just looking ahead, change the copy, not the original
    if action == -1:
        board = original_board
    else:
        # copy everything except the agents. use greedy for all 4
        copy_players = [p.position for p in original_board.players]
        copy_foods =   [f.position for f in original_board.foods]
        copy_rounds =  original_board.round
        copy_scores =  original_board.score[:]
        board = State(copy_players, copy_foods, copy_rounds, copy_scores)
    # get new desired positions
    desired_spaces = [[0, 0]] * num_players
    # get players' desired moves
    desired_spaces[0] = board.players[0].move(board)
    if action == -1:
        desired_spaces[1] = board.players[1].move(board)
    else:
        desired_spaces[1] = get_desired_space_from_action(board.players[0].position, action)
    desired_spaces[2] = board.players[2].move(board)
    desired_spaces[3] = board.players[3].move(board)
    # physically move players, award points, and track eaten foods
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
        # check for walls
        overboard = False
        if (desired_spaces[i][0] < 0 or desired_spaces[i][0] >= board_size or
            desired_spaces[i][1] < 0 or desired_spaces[i][1] >= board_size):
            overboard = True
        # award points
        if ate_food:
            board.score[i] += 2
            if collided:
                board.score[i] -= 1
        # move the player (if not overboard)
        if not collided and not overboard:
            board.players[i].position = desired_spaces[i]
        elif not overboard:
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
    return board


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
    print("8dqn")

    # initialize the board
    board = State()
    old_scores = [0, 0, 0, 0]
    # play some rounds
    # t = time.time()
    if save_data:
        f = open("data.json", "w+")
        f.write("[")
    for r in range(0, rounds):
        if save_data:
            positions = [p.position for p in board.players]
            targets = [[f[0].position, f[1]] for f in [p.pick_target(board) for p in board.players]]
            old_scores = [s for s in board.score]
            board_mtx = np.zeros([board_size, board_size, 4], dtype=int)
            # one-hot encoding: [player, ally, enemy, food]
            for i in range(num_players):
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
        board = play_round(board)
        if save_data:
            actions = [p.action for p in board.players]
            new_scores = [s for s in board.score]
            scores = [0] * num_players
            for s in range(num_players):
                scores[s] = new_scores[s] - old_scores[s]
            data_dict = {'board': board_mtx.tolist(),
                         'positions': positions,
                         'targets': targets,
                         'actions': actions,
                         'scores': scores}
            data = json.dumps(data_dict)
            f.write(data)
            if r != rounds-1:
                f.write(", ")
            else:
                f.write("]")
                f.close()
        if board.round%10000 == 0:
            new_scores = [board.score[i] - old_scores[i] for i in range(4)]
            old_scores = board.score[:]
            print("All scores:", new_scores)
    if display:
        display_board(board)
        input()
    print("All scores:", board.score)
    print("Team scores:", [board.score[0]+board.score[1], board.score[2]+board.score[3]])
    # print(time.time() - t, "seconds")


if __name__ == "__main__":
    main()
