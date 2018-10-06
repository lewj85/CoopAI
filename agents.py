from game import board_size, num_players

action_dict = {
    0: "up",
    1: "right",
    2: "down",
    3: "left",
    4: "stay"
}


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
        self.target = None
        self.action = None
        self.stuck = False

    def move(self, board):
        sess = tf.Session()
        saver = tf.train.import_meta_graph('./simple_simple_clean_copy1.meta')
        ckpt = tf.train.get_checkpoint_state('./')
        # saver.restore(sess, tf.train.latest_checkpoint('./'))
        saver.restore(sess, ckpt.model_checkpoint_path)
        model = load_model('./dqn3nnowood_332_99_0001_1_sgd_1000_no_a95d_filt25_q1_bt_continue_final.h5')
        return desired_position