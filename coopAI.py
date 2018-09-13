import random
import time

# global variables
BOARD_SIZE = 12
NUM_PLAYERS = 4
ROUNDS = 100000
DISPLAY = False
STOP = False
# format = [row, col]


###################################################
# AGENTS
###################################################
def greedy_clockwise(i, occupied_spaces, score):
    player = occupied_spaces[i]
    players = occupied_spaces[:NUM_PLAYERS]
    foods = occupied_spaces[NUM_PLAYERS:]

    # find closest food
    closest_food = [BOARD_SIZE * 2, [0, 0]]
    j = NUM_PLAYERS
    for food in foods:
        # manhattan distance
        # NOTE: does not take into account obstacles (other players)
        distance = abs(player[0] - food[0]) + abs(player[1] - food[1])
        if distance < closest_food[0]:
            closest_food = [distance, food, j]
        j += 1
    # move toward it in clockwise preference: up > right > down > left
    moved = False
    # check up
    if player[0] > closest_food[1][0]:
        # make sure you stay on the board
        if player[0]-1 >= 1:
            new_position = [player[0]-1, player[1]]
            # if not occupied by a player, move
            if new_position not in players:
                occupied_spaces[i] = new_position
                moved = True
            # if food is eaten, update score and move food
            if new_position in foods:
                score[i % 2 == 1] += 1
                occupied_spaces[closest_food[2]] = find_empty_spot(occupied_spaces)
    # check right
    if not moved and player[1] < closest_food[1][1]:
        # make sure you stay on the board
        if player[1]+1 <= BOARD_SIZE:
            new_position = [player[0], player[1]+1]
            # if not occupied by a player, move
            if new_position not in players:
                occupied_spaces[i] = new_position
                moved = True
            # if food is eaten, update score and move food
            if new_position in foods:
                score[i % 2 == 1] += 1
                occupied_spaces[closest_food[2]] = find_empty_spot(occupied_spaces)
    # check down
    if not moved and player[0] < closest_food[1][0]:
        # make sure you stay on the board
        if player[0]+1 <= BOARD_SIZE:
            new_position = [player[0]+1, player[1]]
            # if not occupied by a player, move
            if new_position not in players:
                occupied_spaces[i] = new_position
                moved = True
            # if food is eaten, update score and move food
            if new_position in foods:
                score[i % 2 == 1] += 1
                occupied_spaces[closest_food[2]] = find_empty_spot(occupied_spaces)
    # check left
    if not moved and player[1] > closest_food[1][1]:
        # make sure you stay on the board
        if player[1]-1 >= 1:
            new_position = [player[0], player[1]-1]
            # if not occupied by a player, move
            if new_position not in players:
                occupied_spaces[i] = new_position
                # moved = True
            # if food is eaten, update score and move food
            if new_position in foods:
                score[i % 2 == 1] += 1
                occupied_spaces[closest_food[2]] = find_empty_spot(occupied_spaces)

    # if stuck against a wall and can't move, stand still (ie. don't do anything)

    return occupied_spaces


def coop_ai():
    pass


###################################################
# GAME
###################################################
def initialize_board():
    # initialize players
    p1 = [3, 3]  # team 1
    p2 = [3, BOARD_SIZE-2]  # team 2
    p3 = [BOARD_SIZE-2, BOARD_SIZE-2]  # team 1
    p4 = [BOARD_SIZE-2, 3]  # team 2
    occupied_spaces = [p1, p2, p3, p4]

    # initialize food
    f1 = find_empty_spot(occupied_spaces)
    occupied_spaces.append(f1)
    f2 = find_empty_spot(occupied_spaces)
    occupied_spaces.append(f2)
    f3 = find_empty_spot(occupied_spaces)
    occupied_spaces.append(f3)
    f4 = find_empty_spot(occupied_spaces)
    occupied_spaces.append(f4)

    return occupied_spaces


def find_empty_spot(occupied_spaces):
    while True:
        spot = [random.randint(1, BOARD_SIZE), random.randint(1, BOARD_SIZE)]
        if spot not in occupied_spaces:
            return spot


def play_round(occupied_spaces, score):
    for i in range(NUM_PLAYERS):
        occupied_spaces = greedy_clockwise(i, occupied_spaces, score)

    return occupied_spaces, score


def display_board(occupied_spaces, score):
    print("\n\n\n\n" + ". " * (BOARD_SIZE+1))
    for row in range(1, BOARD_SIZE+1):
        row_string = "" + (". " * (BOARD_SIZE+1))
        i = 1
        for player in occupied_spaces[:NUM_PLAYERS]:
            if player[0] == row:
                row_string = row_string[:player[1]*2-1] + str(i) + row_string[player[1]*2:]
            i += 1
        for food in occupied_spaces[NUM_PLAYERS:]:
            if food[0] == row:
                row_string = row_string[:food[1]*2-1] + "o" + row_string[food[1]*2:]
        print(row_string)
    print("score:", score)


###################################################
# MAIN
###################################################
def main():
    # initialize the board
    occupied_spaces = initialize_board()

    # initialize scores
    score = [0, 0]

    if DISPLAY:
        display_board(occupied_spaces, score)
        if STOP:
            input()
        else:
            time.sleep(0.5)

    # play some rounds
    for i in range(ROUNDS):
        occupied_spaces, score = play_round(occupied_spaces, score)

        if DISPLAY:
            display_board(occupied_spaces, score)
            if STOP:
                input()
            else:
                time.sleep(0.5)

    print("Final score:", score)


if __name__ == "__main__":
    main()
