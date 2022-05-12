from time import perf_counter
import cv2
import multiprocessing as mp

from tetris import Tetris
from mcts import MctsNode, MctsTree
from tqueue import TQueue
from evaluator import Evaluator
import utility as util


class MctsPlay:
    def __init__(self,
                 *,
                 simulation_no,
                 eval_policy,
                 rollout_depth,
                 c_param,
                 max_move_no: int = 1000000,
                 time_budget: float = None):
        self.simulation_no = simulation_no
        self.eval_policy = eval_policy
        self.rollout_depth = rollout_depth
        self.c_param = c_param
        self.max_move_no = max_move_no
        self.time_budget = time_budget

    def update_stats(self, s: dict, d: dict) -> dict:
        for k, v in d.items():
            s[k] += v
        return s

    def play(self, game_no: int, render: bool = False, proc_id: int = 1) -> dict:
        window_name = f'Tetris_{proc_id}'
        if render:
            width, height = 300, 650
            cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(window_name, width, height)
            cv2.moveWindow(window_name, (proc_id - 1) * (width + 20) + 10, 215)

        queue_length = (self.simulation_no + self.rollout_depth) * self.max_move_no
        tetris = Tetris(20, 10)
        stats = tetris.stats()
        for g in range(game_no):
            tetris.reset()
            t_queue = TQueue(queue_length)
            evaluator = Evaluator(self.eval_policy, self.rollout_depth, t_queue)
            MctsNode.evaluator = evaluator
            MctsNode.t_queue = t_queue
            MctsNode.c_param = self.c_param

            tetris.spawn_new_tetromino(t_queue.get(0))
            state = tetris.get_state_for_mcts()
            root = MctsNode(state=state)
            tree = MctsTree()
            while not tetris.game_over and self.max_move_no - tetris.number_of_tetrominos_placed > 0:
                tetris.spawn_new_tetromino(t_queue.get(0))
                tree.set_root(root)

                child_node = tree.best_action(simulation_no=self.simulation_no, time_budget=self.time_budget)
                action = child_node.parent_action
                column = action % 10
                rotation = action // 10

                tetris.drop_one_tetromino(column, rotation, render=render, window_name=window_name, wait=1)

                root.parent = None
                root.parent_action = None
                MctsTree.decrement_indices(child_node)
                root = child_node
                t_queue.remove_first()

            stats = self.update_stats(stats, tetris.stats())

        if render:
            cv2.waitKey(2000)
        return stats.copy()


def sum_stats(stats: list) -> dict:
    summed_stats = stats[0].copy()
    if len(stats) > 1:
        for d in stats[1:]:
            for k, v in d.items():
                summed_stats[k] += v
    return summed_stats


def str_from_stats(args: dict, stats: dict, time: float, game_no: int, cores: int) -> str:
    info = (f'time: {time:.4f}s\n'
            f'avg_move_time: {(cores * time / stats["game_length"]):.4f}s\n')
    info += f'\ngame_no: {game_no}\n'
    info += '\n'.join([f'{k}: {v}' for k, v in args.items()]) + '\n'

    info += '\n' + '\n'.join([f'AVG_{k.upper()}: {(v / game_no):.2f}' for k, v in stats.items()]) + '\n'

    tmp = info.split('\n')
    tmp2 = []
    for line in tmp:
        t = line.split(' ')
        if len(t) == 1:
            t = [' ', ' ']
        tmp2.append(t)

    info = util.pretty_list_of_strings(tmp2, csep='', align='left')
    info = ('\n'.join(info))

    return info


def divide_game_no(game_no: int, cores: int) -> list:
    integer, remainder = divmod(game_no, cores)
    args = [integer for i in range(cores)]
    for i in range(remainder):
        args[i] += 1
    return args


def play(mctsgame: MctsPlay, game_no: int, cores: int = mp.cpu_count(), render: bool = False) -> str:
    gnos = divide_game_no(game_no, cores)
    args = [[gnos[i], render, i + 1] for i in range(len(gnos))]

    start = perf_counter()
    if cores == 1:
        result = [mctsgame.play(*args[0])]
    else:
        with mp.Pool(processes=cores) as pool:
            result = pool.starmap(mctsgame.play, args)
    time = perf_counter() - start

    stats = sum_stats(result)
    info = str_from_stats(args=mctsgame.__dict__, stats=stats, time=time, game_no=game_no, cores=cores)

    return info
