import coopAI

BOARD_SIZE = coopAI.BOARD_SIZE
NUM_PLAYERS = coopAI.NUM_PLAYERS


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
