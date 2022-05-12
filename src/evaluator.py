import numpy as np
from numba import njit
from keras.models import load_model

from tetris import Tetris
from utility import get_legal_actions, is_game_over
from tetris import get_max_height, calc_no_of_holes, insert_tetromino_in_lowest_row, insert_tetromino, calc_no_of_cleared_lines
import global_vars as gv
from tqueue import TQueue


class Evaluator:
    def __init__(self, policy: str, depth: int = 0, tqueue: TQueue = None):
        self.policy = policy
        self.depth = depth
        self.ev = None
        self.tqueue = tqueue
        self.ev_n = EvaluatorNeural()
        self.ev_h = EvaluatorHeuristic()
        self.ev_r = EvaluateRollout(self.depth, tqueue)

    def evaluate(self, state: np.ndarray, t_index: int = None) -> (float, np.ndarray):
        if self.policy == 'neural':
            return self.ev_n.evaluate(state)
        elif self.policy == 'heuristic':
            return self.ev_h.evaluate(state)
        elif self.policy == 'rollout':
            return self.ev_r.evaluate(state, t_index)


class EvaluateRollout:
    def __init__(self, depth: int, tqueue: TQueue):
        self.game = Tetris(20, 10)
        self.depth = depth
        self.tqueue = tqueue

    def evaluate(self, state: np.ndarray, t_index: int) -> (float, np.ndarray):
        if is_game_over(state):
            return 0, None

        local_state = state.copy()
        self.game.reset()
        self.game.set_state(local_state)
        for i in range(t_index, t_index + self.depth):
            if self.game.game_over:
                return 0, None
            new_t = self.tqueue.get(i)
            self.game.spawn_new_tetromino(new_t)
            local_state = self.game.get_state()
            actions = get_legal_actions(local_state)
            action = actions[np.random.randint(len(actions))]
            column = action % 10
            rotation = action // 10
            self.game.drop_one_tetromino(column, rotation)

        q = 1 / (1 + np.exp(-self.game.get_score()))
        return q, None


class EvaluatorHeuristic:
    def __init__(self):
        pass

    def evaluate(self, state: np.ndarray) -> (float, np.ndarray):
        if is_game_over(state):
            return 0, None

        actions = get_legal_actions(state)
        height, width = 24, 10
        board = state[:-7].reshape(height, width)
        ct_type = np.where(state[-7:] == 1)[0][0]
        max_height_vector = np.zeros(width, dtype='int8')

        q_vector = np.zeros(len(actions), dtype='float32')
        for i in range(len(actions)):
            column = actions[i] % 10
            rotation = actions[i] // 10
            offset = gv.offsets[ct_type][rotation]
            ct_shape = gv.tetromino_shape[ct_type][rotation]
            q_vector[i] = self.make_move_eval(board, column, offset, ct_shape, max_height_vector)
        return np.max(q_vector), q_vector

    @staticmethod
    @njit(cache=True)
    def make_move_eval(board: np.ndarray, column: int, offset: np.ndarray, ct_shape: np.ndarray, max_height_vector: np.ndarray) -> float:
        reward = 0
        _ = get_max_height(max_height_vector, board)
        row = insert_tetromino_in_lowest_row(board, column, offset, max_height_vector, ct_shape)
        cleared_lines = calc_no_of_cleared_lines(board, row)
        max_height = get_max_height(max_height_vector, board) - cleared_lines

        if max_height <= board.shape[0] - 4:
            # reward += [0, 20, 50, 150, 600][cleared_lines]
            # reward -= 100 * calc_no_of_holes(board, max_height_vector)
            reward += [0, -20, -50, -150, 600][cleared_lines]
            reward -= 100 * calc_no_of_holes(board, max_height_vector)
            reward -= 3 * max_height
            reward = 1 / (1 + np.exp(-reward / 100))
        else:
            reward -= 1000
        insert_tetromino(board, row, column, -1 * ct_shape)
        return reward


class EvaluatorNeural:
    nn_model = load_model('nn_model\\ttrs_2490_model.h5', compile=False)

    def __init__(self):
        pass

    @staticmethod
    def evaluate(state: np.ndarray) -> (float, np.ndarray):
        if is_game_over(state):
            return 0, None

        t = state[-7:]
        state_tmp = np.minimum(1, state[:200])
        state_tmp = np.concatenate((state_tmp, t))
        state_tmp = state_tmp.astype('int8')

        q_vector = EvaluatorNeural.nn_model.predict(np.array([state_tmp]))[0]

        ct_type = np.where(state[-7:] == 1)[0][0]  # average over symmetrical moves
        symmetries = gv.symmetrical_actions[ct_type]
        for sym in symmetries:
            EvaluatorNeural.average_over_symmetries(q_vector, np.array(sym))

        actions = get_legal_actions(state)  # map q_vector from neural net to legal actions
        q_vector = q_vector[actions]

        height, width = 24, 10
        board = state[:-7].reshape(height, width)
        ct_type = np.where(state[-7:] == 1)[0][0]
        max_height_vector = np.zeros(width, dtype='int8')
        for i in range(len(actions)):
            column = actions[i] % 10
            rotation = actions[i] // 10
            offset = gv.offsets[ct_type][rotation]
            ct_shape = gv.tetromino_shape[ct_type][rotation]
            q_vector[i] += EvaluatorNeural.make_move_eval(board, column, offset, ct_shape, max_height_vector)

        q_vector = EvaluatorNeural.normalize(q_vector)
        return np.max(q_vector), q_vector

    @staticmethod
    @njit(cache=True)
    def average_over_symmetries(q_vector: np.ndarray, sym: np.ndarray) -> None:
        q_vector[sym[0]] = np.mean(q_vector[sym])

    @staticmethod
    @njit(cache=True)
    def normalize(data: np.ndarray) -> np.ndarray:
        mean = 232.24489522685028
        stddev = 113.48066416801656
        data = (data - mean) / stddev
        data = 1 / (1 + np.exp(-data))
        return data

    @staticmethod
    @njit(cache=True)
    def make_move_eval(board: np.ndarray, column: int, offset: np.ndarray, ct_shape: np.ndarray, max_height_vector: np.ndarray) -> float:
        _ = get_max_height(max_height_vector, board)
        row = insert_tetromino_in_lowest_row(board, column, offset, max_height_vector, ct_shape)
        cleared_lines = calc_no_of_cleared_lines(board, row)
        max_height = get_max_height(max_height_vector, board) - cleared_lines

        if max_height <= board.shape[0] - 4:
            insert_tetromino(board, row, column, -1 * ct_shape)
            return [0, 40, 100, 300, 1200][cleared_lines]
        else:
            insert_tetromino(board, row, column, -1 * ct_shape)
            return -2000
