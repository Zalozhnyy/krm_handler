import json
import os
import sys
import json
import time

from itertools import chain
from typing import List
from collections import defaultdict

import numpy as np

sys.path.append(os.path.dirname(__file__))

import parse_T7

SAVE_DIR: str = None
KRM_DIR: str = None


def execution_time(func):
    def wrapper(*args, **kwargs):
        t = time.time()
        result = func(*args, **kwargs)
        print(f'{func.__name__} done in {time.time() - t:.2f} seconds')
        return result

    return wrapper


def mirror_grid(grid):
    return [grid[i] * -1 * 1.e3 for i in range(len(grid) - 1, 0, -1)] + list(map(lambda x: x * 1.e3, grid))


class GrdFileBuilder:
    _UNIFORM_PATTERN = '{COORD_LEFT:.12g}\n' \
                       '{STEP_LEFT:.12g}\n' \
                       '{STEP_RIGHT:.12g}\n' \
                       '{COEF:.12g}\n' \
                       '{POINTS_COUNT:d}\n' \
                       '1\n' \
                       '1\n' \
                       '0\n'

    _AXE_HEADER = '{AXE_NAME}\n' \
                  '{AXE_NAME}\n' \
                  '{SEGMENTS_COUNT}\n' \
                  '{POINTS_COUNT}\n' \
                  '10.0\n'

    _HEADER_PATTERN = '<FULL>\n' \
                      '1\n' \
                      '<EQUAL (0-not equal 1-equal)>\n' \
                      '1\n' \
                      'X\n' \
                      '{X_COUNT}\n' \
                      '{X_GRID}\n' \
                      'Y\n' \
                      '{Y_COUNT}\n' \
                      '{Y_GRID}\n' \
                      'Z\n' \
                      '{Z_COUNT}\n' \
                      '{Z_GRID}\n' \
                      'T\n' \
                      '{T_COUNT}\n' \
                      '{T_GRID}\n' \
                      'ENDGRID\n'

    def __init__(self, grid: List[float], time: List[float]):
        self._row_stack = []
        self._grid = grid
        self._time_grid = time

    def _create_head(self):
        count = len(self._grid)
        grid = ' '.join(f"{val:.4f}" for val in self._grid)

        self._row_stack.append(
            self._HEADER_PATTERN.format(
                X_COUNT=count,
                X_GRID=grid,
                Y_COUNT=count,
                Y_GRID=grid,
                Z_COUNT=count,
                Z_GRID=grid,
                T_COUNT=len(self._time_grid),
                T_GRID=' '.join(f"{val:.6E}" for val in self._time_grid),
            )
        )

    def _coord(self, grid: List[float]):

        body = []

        for i in range(len(grid) - 1):
            body.append(
                self._UNIFORM_PATTERN.format(
                    COORD_LEFT=grid[i],
                    STEP_LEFT=grid[i + 1] - grid[i],
                    STEP_RIGHT=grid[i + 1] - grid[i],
                    COEF=1,
                    POINTS_COUNT=2
                )
            )

        body.append(f"{grid[-1]:.12g}\n")

        return ''.join(body)

    def _create_axes(self):

        coords = self._coord(self._grid)

        for axe in ('X', 'Y', 'Z'):
            self._row_stack.append(
                self._AXE_HEADER.format(
                    AXE_NAME=axe,
                    SEGMENTS_COUNT=len(self._grid) - 1,
                    POINTS_COUNT=len(self._grid)
                )
            )

            self._row_stack.append(coords)

        # T axe
        self._row_stack.append(
            self._AXE_HEADER.format(
                AXE_NAME="T",
                SEGMENTS_COUNT=len(self._time_grid) - 1,
                POINTS_COUNT=len(self._time_grid)
            )
        )

        self._row_stack.append(self._coord(self._time_grid))

    def create_grid_file(self, file_name: str):
        self._create_head()
        self._create_axes()

        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(''.join(self._row_stack))


class GreaterGridBuilder:

    def create_grid(self, local_grids, times):

        all_local_grids = set(chain(*[item[1] for item in local_grids]))
        all_local_grids = sorted(list(all_local_grids))

        l = 0
        out = [all_local_grids[0]]
        barrier = all_local_grids[1] - all_local_grids[0]

        for r in range(1, len(all_local_grids)):

            if all_local_grids[r] - all_local_grids[l] > barrier:
                barrier = all_local_grids[r] - out[-1]
                out.append(all_local_grids[r])
                l = r

        grid = mirror_grid(out)

        grid_maker = GrdFileBuilder(grid, times)
        grid_maker.create_grid_file(os.path.join(SAVE_DIR, '..', f'KRM_FULL_GRID.GRD'))


def merge_grid(grid_list: np.ndarray, relative_barrier: float) -> List:
    raw_data = np.unique(np.concatenate(grid_list))
    raw_data.sort()

    output = [raw_data[0], raw_data[1]]
    last_distance = raw_data[1] - raw_data[0]
    prev = raw_data[1]

    for i in range(2, len(raw_data) - 1):

        current_distance = raw_data[i] - prev

        if current_distance > last_distance * relative_barrier:
            output.append(raw_data[i])
            prev = raw_data[i]

    output.append(raw_data[-1])

    return output


def get_grids_from_batch(batch: List) -> np.ndarray:
    return np.array([item[1] for item in batch], dtype=object)


class GridMerger:
    def __init__(self):
        print("start reading setka.txt")
        self._data = read_time_grid(os.path.join(KRM_DIR, 'setka.txt'))
        print("setka.txt reading done")

        self._max_grid_count: int = 0
        self._diff = 0.015

    def _recursive_merge(self, l: int, r: int, diff: float, data: List, results: List):

        if l == r:
            results.append(data[r])
            # raise Exception("Too small max grid count")
            return

        tmp_merge = merge_grid(get_grids_from_batch(data[l:r]), diff)

        if len(tmp_merge) < self._max_grid_count:
            time = data[r][0]
            results.append([time, tmp_merge])
            return

        else:
            m = (l + r) // 2
            self._recursive_merge(l, m - 1, diff, data, results)
            self._recursive_merge(m, r, diff, data, results)

    def _split_data(self):
        start_batch, filtered_batch = [], []

        for time, arr in self._data:
            self._max_grid_count = max(self._max_grid_count, len(arr))
            if time > 1e-6:
                filtered_batch.append([time, np.array(arr)])
            else:
                start_batch.append([time, np.array(arr)])

        return start_batch, filtered_batch

    @execution_time
    def merge_grids(self) -> List:
        start_batch, filtered_batch = self._split_data()

        merged = [[1e-6, merge_grid(get_grids_from_batch(start_batch), 0.05)]]
        self._max_grid_count = self._max_grid_count * 2

        diff = 0.1
        self._recursive_merge(0, len(filtered_batch) - 1, diff, filtered_batch, merged)
        print(f'{len(filtered_batch)} merged to {len(merged)}')

        return merged


def main():
    prepare_project()
    # a = transform_data()
    # times_grid = read_time_grid(os.path.join(KRM_DIR, 'W88_T7_S1.txt'))
    # times = [item[0] for item in times_grid]

    a = GridMerger().merge_grids()

    start_grid = merge_grid(get_grids_from_batch(a), 0.05)

    grids = [[0, start_grid]] + a
    times = [item[0] for item in grids]

    json_header = []

    for time, grid in grids:
        gr = mirror_grid(grid)

        grid_maker = GrdFileBuilder(gr, times)
        grid_maker.create_grid_file(os.path.join(SAVE_DIR, f'GRID_ITERATION_{time}.GRD'))
        json_header.append({'file': f'GRID_ITERATION_{time}.GRD', 'time': time})
        print(f"Создан файл GRID_ITERATION_{time}.GRD")

    with open(os.path.join(SAVE_DIR, f'grid_headers.json'), 'w', encoding='utf-8') as f:
        json.dump(json_header, f, indent=4)

    print(f'Количество созданных файлов: {len(a)}')


@execution_time
def read_time_grid(path: str):
    data = []

    with open(path, 'r', encoding='utf-8') as f:

        line = f.readline().strip().split()

        while True:
            if not line:
                break

            time, n = [float(l) for l in line]
            time *= 1e-6
            n = int(n)
            tmp = (time, [])

            try:
                line = f.readline().strip().split()
                while len(line) == 1:
                    tmp[1].append(float(line[0]))
                    line = f.readline().strip().split()
            except EOFError:
                break

            if len(tmp[1]) != n:
                print(f'WARNING time={time}  n={n} real values={len(tmp[1])}')
                continue

            if len(data) == 0:
                data.append(tmp)
                continue

            if data[-1][0] < tmp[0]:
                data.append(tmp)

    return data


def transform_data():
    data = parse_T7.get_t7_data(KRM_DIR)

    RK_I = data['header'].index("RK(I)")

    array = []
    for key, item in data['data'].items():
        array.append(
            [float(key), [i[RK_I] for i in item]]
        )
    return array


def prepare_project():
    global SAVE_DIR, KRM_DIR

    projectfilename = r'D:\remp_projects\mhd_krmkrm_full10\koverna.PRJ'

    try:
        path = os.path.normpath(projectfilename)
    except Exception:
        path = input("Введите путь к папке проекта: ")

    if not os.path.exists(path):
        print(f'Не найдена папка {path}')
        return

    path = os.path.dirname(path)

    assert os.path.exists(os.path.join(path, 'project_krm'))
    assert os.path.exists(os.path.join(path, 'project_krm', 'Zas.txt'))
    assert os.path.exists(os.path.join(path, 'project_krm', 'W88_T7_RF.TXT'))
    assert os.path.exists(os.path.join(path, 'project_krm', 'W88_T7.txt'))
    assert os.path.exists(os.path.join(path, 'project_krm', 'W88_T7_S1.txt'))

    SAVE_DIR = os.path.join(path, "krm_grids")
    KRM_DIR = os.path.join(path, "project_krm")

    try:
        os.mkdir(SAVE_DIR)
    except FileExistsError:
        pass


if __name__ == '__main__':
    main()

