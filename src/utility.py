import numpy as np
from numba import njit

from tetris import get_max_height, insert_tetromino_in_lowest_row, clear_lines
import global_vars as gv


def get_legal_actions(state: np.ndarray) -> list:
    ct_type = np.where(state[-7:] == 1)[0]
    try:
        ct_type = ct_type[0]
    except TypeError:
        return []
    else:
        return gv.legal_actions[ct_type]


@njit(cache=True)
def is_game_over(state: np.ndarray) -> bool:
    for column in range(-47, -37):
        if state[column] != 0:
            return True
    return False


@njit(cache=True)
def __next_state_help(state: np.ndarray, column: int, offset: np.ndarray, ct_shape: np.ndarray, next_t_type: int) -> np.ndarray:
    state = state.copy()
    state = state[:-7].reshape(24, 10)
    max_heights = np.zeros(state.shape[1], dtype='int8')

    _ = get_max_height(max_heights, state)
    row = insert_tetromino_in_lowest_row(state, column, offset, max_heights, ct_shape)
    _ = clear_lines(state, row)

    state = state.flatten()
    t = np.zeros(7)
    t[next_t_type] = 1
    state = np.concatenate((state, t))
    state = state.astype('int8')
    return state


def get_next_state(state: np.ndarray, next_t_type: int, action: int) -> np.ndarray:
    ct_type = np.where(state[-7:] == 1)[0][0]
    column = action % 10
    rotation = action // 10
    offset = gv.offsets[ct_type][rotation]
    ct_shape = gv.tetromino_shape[ct_type][rotation]

    return __next_state_help(state, column, offset, ct_shape, next_t_type)


def pretty_list_of_strings(data: list, col_names: tuple = (), csep: str = ' ', rsep: str = '', align='left') -> list:
    col_widths = []
    if len(col_names) != 0:
        tmp = data + [col_names]
    else:
        tmp = data
    for i in range(len(tmp[0])):
        col = [str(sub[i]) for sub in tmp]
        col_widths.append(len(max(col, key=len)))

    result = []
    if col_names != ():
        result.append(''.join([f' {csep} '.join([col_names[i].ljust(col_widths[i]) for i in range(len(col_widths))])]))

    if rsep != '':
        tmp = f'{rsep}{csep}{rsep}'
        rsep = tmp.join([width * rsep for width in col_widths])
    else:
        result.append(''.join([width * rsep for width in col_widths]))

    for row in data:
        if rsep != '':
            result.append(rsep)
        new_line = [str(row[col]).ljust(col_widths[col]) for col in range(len(row))]
        if align == 'right':
            new_line = [str(row[col]).rjust(col_widths[col]) for col in range(len(row))]
        elif align == 'center':
            new_line = [str(row[col]).center(col_widths[col]) for col in range(len(row))]
        new_line = f' {csep} '.join(new_line)
        result.append(new_line)
    return result
