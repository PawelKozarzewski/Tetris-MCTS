from mcts_play import MctsPlay, play
from datetime import datetime
import os
import itertools
import multiprocessing as mp
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # turns off keras' warning about not supporting AVX


def get_log_filename(i: int, args: dict):
    n = str(i).zfill(3)
    e = args['eval_policy'][0].upper()
    s = args['simulation_no']
    c = args['c_param']
    d = args['rollout_depth']
    m = args['max_move_no']
    filename = f'{n}_{e}_{s}sim_{c}c_{d}dep_{m}mo.txt'
    return filename


def generate_args(var: dict, const: dict = None, ) -> list:
    keys, values = zip(*var.items())
    d_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    if const is None:
        return d_permutations
    result = [{**const, **d} for d in d_permutations]
    return result


def get_max_file_index(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
    max_index = 0
    for f in os.listdir(path):
        if f.endswith(".txt"):
            max_index = max(max_index, int(f.split('_')[0]))
    return max_index


def play_test_log(argslist: list, cores: int = mp.cpu_count(), render: bool = False, log=True, path: str = ''):
    print(f'NEW ARGUMENT BATCH ({len(argslist)})')
    print('*' * len(str(argslist[0])))
    print(*argslist, sep='\n')
    print('*' * len(str(argslist[0])))

    file_index = get_max_file_index(path) + 1

    batch_start = datetime.now()
    for i in range(1, len(argslist) + 1):
        args = argslist[i - 1]
        print(f'______________________________')
        print(f'Args set #: {i}/{len(argslist)}')
        print(args)
        print('...')

        start_time = datetime.now()
        game_no = args.pop('game_no')
        mg = MctsPlay(**args)
        info = play(mg, game_no=game_no, cores=cores, render=render)
        end_time = datetime.now()

        if log:
            filename = get_log_filename(file_index, args)
            tmp = f'{path}\\{filename}'
            with open(tmp, 'w') as f:
                f.write((f'start_time: {start_time}\n'
                         f'end_time:   {end_time}\n'))
                f.write(info)
            print(f'Saved to: {tmp}')
        file_index += 1

        print(f'Finished in: {datetime.now() - start_time}')
        print()
    print(f'FINISHED ARGUMENT BATCH ({len(argslist)}) IN {datetime.now() - batch_start}\n\n')


if __name__ == '__main__':
    args = {'game_no': [4], 'eval_policy': ['heuristic'], 'rollout_depth': [0], 'max_move_no': [1000],
            'simulation_no': [1000], 'c_param': [0.0], 'time_budget': [0.25]}
    argslist = generate_args(args)
    play_test_log(argslist[0:], cores=1, render=True, log=True, path=f'results\\data')
