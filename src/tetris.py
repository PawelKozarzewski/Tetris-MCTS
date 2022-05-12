import numpy as np
import cv2
from numba import njit
from collections import OrderedDict

import global_vars as gv


@njit(cache=True)
def row_to_drop(column: int, offset: np.ndarray, max_height: np.ndarray) -> int:
    col0 = column
    row = -1
    for i in range(offset.shape[0]):
        row = max(row, max_height[col0 + i] + offset[i])
    return row


@njit(cache=True)
def insert_tetromino(board: np.ndarray, row: int, column: int, ct_shape: np.ndarray) -> None:
    for i in range(ct_shape.shape[0]):
        for j in range(ct_shape.shape[1]):
            board[row + i][column + j] += ct_shape[i][j]


@njit(cache=True)
def get_board_with_ct(board: np.ndarray, row: int, column: int, ct_shape: np.ndarray) -> np.ndarray:
    c_board = np.copy(board)
    for i in range(ct_shape.shape[0]):
        for j in range(ct_shape.shape[1]):
            c_board[row + i][column + j] += ct_shape[i][j]
    return c_board


@njit(cache=True)
def fill_with_colors(img: np.ndarray, c_board: np.ndarray, tetro_colors: np.ndarray) -> None:
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            img[x][y] = tetro_colors[c_board[x][y]]


@njit(cache=True)
def brighten_cleared(img: np.ndarray, c_board: np.ndarray, tetro_colors: np.ndarray, cleared_indices: np.ndarray) -> None:
    factor = 0.5
    for x in cleared_indices:
        for y in range(0, img.shape[1]):
            color = tetro_colors[c_board[x][y]]
            color = np.array([255, 255, 255]) - (np.array([255, 255, 255]) - color) * (1 - factor)
            img[x][y] = color


@njit(cache=True)
def get_max_height(result: np.ndarray, board: np.ndarray) -> int:
    max_height = 0
    for col in range(board.shape[1]):
        result[col] = 0
        for row in range(board.shape[0] - 1, -1, -1):
            if board[row][col] != 0:
                result[col] = 1 + row
                break
        max_height = max(max_height, result[col])
    return max_height


@njit(cache=True)
def insert_tetromino_in_lowest_row(board: np.ndarray, column: int, offset: np.ndarray, max_height: np.ndarray, ct_shape: np.ndarray) -> int:
    col0 = column
    row = -1
    for i in range(offset.shape[0]):
        row = max(row, max_height[col0 + i] + offset[i])
    for i in range(ct_shape.shape[0]):
        for j in range(ct_shape.shape[1]):
            board[row + i][col0 + j] += ct_shape[i][j]
    return row


@njit(cache=True)
def calc_no_of_holes(board: np.ndarray, max_height: np.ndarray) -> int:
    holes = 0
    for column in range(board.shape[1]):
        for row in range(max_height[column] - 1):
            if board[row][column] == 0:
                holes += 1
    return holes


@njit(cache=True)
def calc_bumpiness(max_height: np.ndarray) -> int:
    bump_sum = 0
    for i in range(max_height.shape[0] - 1):
        bump_sum += abs(max_height[i] - max_height[i + 1])
    return bump_sum


@njit(cache=True)
def clear_lines(board: np.ndarray, row: int) -> int:
    cleared_lines = 0
    x = row
    height, width = board.shape[0] - 4, board.shape[1]
    while x < row + 4 and cleared_lines < 4:
        non_zero_elements = np.count_nonzero(board[x])
        if non_zero_elements == width:
            board[x:-1] = board[x + 1:]
            board[height + 3].fill(0)
            cleared_lines += 1
            x -= 1
        elif non_zero_elements == 0:
            break
        x += 1
    return cleared_lines


@njit(cache=True)
def calc_no_of_cleared_lines(board: np.ndarray, row: int) -> int:
    cleared_lines = 0
    x = row
    width = board.shape[1]
    while x < row + 4 and cleared_lines < 4:
        non_zero_elements = np.count_nonzero(board[x])
        if non_zero_elements == width:
            cleared_lines += 1
        elif non_zero_elements == 0:
            break
        x += 1
    return cleared_lines


@njit(cache=True)
def calc_no_of_cleared_lines_and_get_indices(board: np.ndarray, row: int) -> (int, np.ndarray):
    cleared_lines = 0
    x = row
    width = board.shape[1]
    cleared_indices = []
    while x < row + 4 and cleared_lines < 4:
        non_zero_elements = np.count_nonzero(board[x])
        if non_zero_elements == width:
            cleared_lines += 1
            cleared_indices.append(x)
        elif non_zero_elements == 0:
            break
        x += 1
    return cleared_lines, np.array(cleared_indices)


class Tetris:
    def __init__(self, height: int, width: int):
        self._height = height
        self._width = width
        self.board = np.zeros((self._height + 4, self._width), dtype='int8')
        self._max_height = np.zeros(self._width, dtype='int8')

        self.empty_value = 0
        self.tetro_values = [1, 2, 3, 4, 5, 6, 7]
        self.tetro_colors = np.array([[192, 192, 192], [49, 199, 239], [90, 101, 173], [239, 121, 33],
                                      [247, 211, 8], [66, 182, 66], [173, 77, 156], [239, 32, 41]])

        self._offsets = gv.offsets
        self._max_right_shift = gv.max_right_shift
        self._tetromino_shape = gv.tetromino_shape

        self.ct_type = None
        self.ct_pos = []
        self.ct_rot = None
        self.bag = []
        self.score = 0
        self.game_over = False
        self.bumpiness = 0
        self.number_of_holes = 0
        self.overall_lines_cleared = np.zeros(4, dtype='int32')
        self.number_of_tetrominos_placed = 0
        self.sum_of_column_heights = 0
        self.reset()

    def reset(self) -> None:
        self.board.fill(0)
        self._max_height.fill(0)
        self.ct_type = None
        self.ct_pos = []
        self.ct_rot = None
        self.bag = np.random.permutation(7)
        self.score = 0
        self.bumpiness = 0
        self.number_of_holes = 0
        self.overall_lines_cleared = np.zeros(4, dtype='int32')
        self.number_of_tetrominos_placed = 0
        self.sum_of_column_heights = 0
        self.game_over = False

    def stats(self) -> dict:
        stats = OrderedDict([('score', self.score),
                             ('game_length', self.number_of_tetrominos_placed),
                             ('1-line', self.overall_lines_cleared[0]),
                             ('2-line', self.overall_lines_cleared[1]),
                             ('3-line', self.overall_lines_cleared[2]),
                             ('4-line', self.overall_lines_cleared[3]),
                             ('total_cleared', int(np.sum(np.multiply(self.overall_lines_cleared, np.array([1, 2, 3, 4])))))])
        return stats.copy()

    def set_state(self, state: np.ndarray) -> None:
        tmp = state[:-7].reshape(self._height + 4, self._width)
        self.board = tmp
        max_height = get_max_height(self._max_height, self.board)
        self.game_over = max_height > self._height

    def get_state(self) -> np.ndarray:
        state = (np.minimum(1, self.board[:self._height, :])).flatten()
        t = np.zeros(7)
        t[self.ct_type] = 1
        state = np.concatenate((state, t))
        state = state.astype('int8')
        return state

    def get_state_for_mcts(self) -> np.ndarray:
        state = self.board.flatten()
        t = np.zeros(7)
        if self.ct_type is not None:
            t[self.ct_type] = 1
        state = np.concatenate((state, t))
        state = state.astype('int8')
        return state

    def get_score(self) -> float:
        return self.score

    def get_cleared_lines_stats(self) -> np.ndarray:
        return self.overall_lines_cleared

    def get_number_of_tetrominos_placed(self) -> int:
        return self.number_of_tetrominos_placed

    def spawn_new_tetromino(self, ct_type: int = None) -> None:
        if ct_type is not None:
            self.ct_type = ct_type
            return
        self.ct_pos = [0, self._width // 2 - 1]
        self.ct_rot = 0
        if len(self.bag) == 0:
            self.bag = np.random.permutation(7)
        self.ct_type = self.bag[0]
        self.bag = self.bag[1:]

    def drop_one_tetromino(self, column: int, rotation: int, render: bool = False, window_name: str = None, wait: int = 1, scale: int = 8) -> None:
        column = min(column, self._max_right_shift[self.ct_type][rotation])
        offset = self._offsets[self.ct_type][rotation]
        ct_shape = self._tetromino_shape[self.ct_type][rotation]

        if not render or not window_name:
            row = insert_tetromino_in_lowest_row(self.board, column, offset, self._max_height, ct_shape)
        else:
            row = row_to_drop(column, offset, self._max_height)

            for i in range(self.board.shape[0] - 4, row - 1, -1):
                c_board = get_board_with_ct(self.board, i, column, ct_shape)
                self.render(window_name, c_board, wait, scale, )
            insert_tetromino(self.board, row, column, ct_shape)

            cleared, cleared_indices = calc_no_of_cleared_lines_and_get_indices(self.board, row)
            if cleared > 0:
                self.render(window_name, self.board, wait, scale, cleared=cleared_indices)

        cleared_lines = clear_lines(self.board, row)
        if cleared_lines > 0:
            self.overall_lines_cleared[cleared_lines - 1] += 1

        max_height = get_max_height(self._max_height, self.board)

        self.score += [0, 40, 100, 300, 1200][cleared_lines]

        self.game_over = max_height > self._height
        self.number_of_tetrominos_placed += 1

        return

    def render(self, window_name: str, c_board: np.ndarray, wait: int, scale: int, cleared: np.ndarray = None) -> None:
        img = np.zeros((c_board.shape[0], c_board.shape[1], 3), dtype='uint8')

        fill_with_colors(img, c_board, self.tetro_colors)

        if cleared is not None:
            if cleared.shape[0] > 0:
                brighten_cleared(img, c_board, self.tetro_colors, cleared)

        img = img[:, :, ::-1]
        img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

        v_border = np.full((img.shape[0], 3), fill_value=0, dtype='uint8')
        h_border = np.full((img.shape[1], 3), fill_value=0, dtype='uint8')
        v_separator = np.full((img.shape[0], 3), fill_value=196, dtype='uint8')
        h_separator = np.full((img.shape[1], 3), fill_value=196, dtype='uint8')

        for x in range(0, img.shape[0], scale):
            img[x, :] = h_separator
        for y in range(0, img.shape[1], scale):
            img[:, y] = v_separator

        img[:, 0] = v_border
        img[:, -1] = v_border
        img[0, :] = h_border
        img[-1, :] = h_border
        img[20 * scale, :] = h_border

        img = img[::-1, :, :]

        score_text = np.full((scale, scale * 10, 3), fill_value=255, dtype='uint8')
        st = Tetris.score_to_image(self.score)
        score_text[1:st.shape[0] + 1, 1:st.shape[1] + 1, :] = st
        img = np.concatenate((img, score_text), axis=0)

        cv2.imshow(window_name, img)
        cv2.waitKey(wait)

    @staticmethod
    @njit(cache=True)
    def score_to_image(score: int) -> np.ndarray:
        text = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                         [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                         [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
        digits = {
            '0': np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 1, 0],
                           [0, 1, 0],
                           [0, 0, 0]]),

            '1': np.array([[1, 1, 0],
                           [1, 1, 0],
                           [1, 1, 0],
                           [1, 1, 0],
                           [1, 1, 0]]),

            '2': np.array([[0, 0, 0],
                           [1, 1, 0],
                           [0, 0, 0],
                           [0, 1, 1],
                           [0, 0, 0]]),

            '3': np.array([[0, 0, 0],
                           [1, 1, 0],
                           [0, 0, 0],
                           [1, 1, 0],
                           [0, 0, 0]]),

            '4': np.array([[0, 1, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [1, 1, 0],
                           [1, 1, 0]]),

            '5': np.array([[0, 0, 0],
                           [0, 1, 1],
                           [0, 0, 0],
                           [1, 1, 0],
                           [0, 0, 0]]),

            '6': np.array([[0, 0, 0],
                           [0, 1, 1],
                           [0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]]),

            '7': np.array([[0, 0, 0],
                           [1, 1, 0],
                           [1, 1, 0],
                           [1, 1, 0],
                           [1, 1, 0]]),

            '8': np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]]),

            '9': np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [1, 1, 0],
                           [0, 0, 0]])
        }
        text = 255 * text

        t = str(score)
        number = np.ones((5, 4 * len(t)))
        number = 255 * number
        for i in range(0, len(t)):
            l = 3 * i + i
            r = 3 * (i + 1) + i
            number[0:, l:r] = 255 * digits[t[i]]

        res = np.ones((5, text.shape[1] + number.shape[1]), dtype='uint8')
        res = 255 * res
        res[:, 0:text.shape[1]] = text
        res[:, text.shape[1]:] = number
        res = np.dstack((res, res, res))
        return res
