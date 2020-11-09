#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import typing
from copy import deepcopy
import matplotlib.pyplot as plt
import time
from itertools import combinations, permutations, repeat, product
from random import shuffle
from datetime import datetime, timedelta


def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")


if is_interactive():
    from IPython.display import clear_output
else:
    # don't bother without put clearing if we're not in IPython
    def clear_output(*args, **kwargs):
        return


def split_path(path):
    path_bookend = np.concatenate([[999], path, [999]])
    starts = np.where(np.diff((path_bookend != 0).astype(int)) == -1)[0]
    ends = np.where(np.diff((path_bookend != 0).astype(int)) == 1)[0]
    segments = []
    for start, end in zip(starts, ends):
        if start > 0:
            left = path[start - 1] + 1
        else:
            left = 1
        if end == len(path):
            right = 10
        else:
            right = path[end]
        segments.append(
            zip(
                repeat(slice(start, end)),
                list(combinations(range(left, right), end - start)),
            )
        )
    return list(product(*segments))


class SudokuCondition:
    """Template for sudoku condition"""

    def test(self, grid: np.ndarray, num: int, row: int, col: int) -> bool:
        """
        Implement testing here, return True if valid
        
        Paramters
        ---------
        grid: np.ndarray
            Grid representing puzzle, should be 9x9 array. Use 0 for unspecified entries
        num: int
            Number to test, between 1 and 9
        row: int
            Row to insert num in, between 0 and 8
        col: int
            Col to insert num in, between 0 and 8
        """
        pass

    def __call__(self, grid: np.ndarray, num: int, row: int, col: int) -> bool:
        """Syntactic sugar to treat class instance as function, see `test()`"""
        return self.test(grid, num, row, col)


class RookCondition(SudokuCondition):
    """Make sure the same number does not exist in the same row or col"""

    def test(self, grid: np.ndarray, num: int, row: int, col: int) -> bool:
        try:
            assert num not in grid[row, :col] and num not in grid[row, col + 1 :]
            assert num not in grid[:row, col] and num not in grid[row + 1 :, col]
        except AssertionError:
            return False
        else:
            return True


class BlockCondition(SudokuCondition):
    """Make sure the same number does not exist in the same 3x3 block"""

    def test(self, grid: np.ndarray, num: int, row: int, col: int) -> bool:
        grid = grid.copy()
        grid[row, col] = 0
        block_row = (row // 3) * 3
        block_col = (col // 3) * 3
        return num not in grid[block_row : block_row + 3, block_col : block_col + 3]


class KingCondition(SudokuCondition):
    """Make sure the same number does not appear a diagonal king move away"""

    def test(self, grid: np.ndarray, num: int, row: int, col: int) -> bool:
        possible_diags = [
            (row - 1, col - 1),
            (row - 1, col + 1),
            (row + 1, col - 1),
            (row + 1, col + 1),
        ]
        possible_diags = [
            x
            for x in possible_diags
            if x[0] >= 0 and x[0] <= 8 and x[1] >= 0 and x[1] <= 8
        ]
        return all((grid[p, q] != num for p, q in possible_diags))


class KnightCondition(SudokuCondition):
    """Make sure no identical numbers a knight move away"""

    def test(self, grid: np.ndarray, num: int, row: int, col: int) -> bool:
        possible_knight = [
            (row - 1, col - 2),
            (row - 2, col - 1),
            (row - 1, col + 2),
            (row - 2, col + 1),
            (row + 1, col - 2),
            (row + 2, col - 1),
            (row + 1, col + 2),
            (row + 2, col + 1),
        ]
        possible_knight = [
            x
            for x in possible_knight
            if x[0] >= 0 and x[0] <= 8 and x[1] >= 0 and x[1] <= 8
        ]
        return all((grid[p, q] != num for p, q in possible_knight))


class ConsecutiveCondition(SudokuCondition):
    """Make sure no consecutive numbers orthogonally"""

    def test(self, grid: np.ndarray, num: int, row: int, col: int) -> bool:
        possible_adjacent = [
            (row, col + 1),
            (row + 1, col),
            (row, col - 1),
            (row - 1, col),
        ]
        possible_adjacent = [
            x
            for x in possible_adjacent
            if x[0] >= 0 and x[0] <= 8 and x[1] >= 0 and x[1] <= 8
        ]
        return all(
            (
                grid[p, q] == 0 or np.abs(grid[p, q] - num) > 1
                for p, q in possible_adjacent
            )
        )


class ComboCondition(SudokuCondition):
    """Combine multiple SudokuCondition classses into one"""

    def __init__(self, *args):
        self.conditions = args

    def test(self, grid: np.ndarray, num: int, row: int, col: int) -> bool:
        try:
            for condition in self.conditions:
                if isinstance(condition, type):
                    assert condition()(grid, num, row, col)
                else:
                    assert condition(grid, num, row, col)
        except AssertionError:
            return False
        else:
            return True


def get_num_order(grid: np.ndarray, partial: bool = False) -> list:
    nums, counts = np.unique(grid, return_counts=True)
    num_order = sorted(range(len(nums)), key=counts.__getitem__, reverse=True)
    num_order = [nums[x] for x in num_order]
    # add numbers not in the grid and filter 0
    num_order = [x for x in num_order if x != 0]
    if not partial:
        num_order += [x for x in range(1, 10) if x not in num_order]
    return num_order


def deduce(arr: np.array) -> np.array:
    if np.count_nonzero(arr) == 8:
        missing_no = (set(range(1, 10)) - set(arr)).pop()
        arr[arr == 0] = missing_no
        return True
    return False


class SudokuSolver:
    """Generic sudoku solver using an arbitrary SudokuCondition ruleset"""

    def __init__(self, grid: np.array, condition: SudokuCondition):
        """
        Parameters
        ----------
        grid: np.ndarray
            Grid representing puzzle, should be 9x9 array. Use 0 for unspecified entries
        condition: SudokuCondition
            SudokuCondition class representing the ruleset
        """
        self.grid = grid
        self.condition = condition
        self.display_time = None

    def candidates(
        self, grid: np.ndarray, row: int, col: int, possibilities_list: list = None
    ) -> list:
        """
        Compute all possiblities for a particular row and col

        Parameters
        ----------
        grid: np.ndarray
            Grid representing puzzle, should be 9x9 array. Use 0 for unspecified entries
        row: int
            Row to insert num in, between 0 and 8
        col: int
            Col to insert num in, between 0 and 8
        possibilities_list: list
            List of integers between 1 and 9, should include all numbers to consider
        """
        if possibilities_list is None:
            possibilities_list = range(1, 10)
        return [
            num for num in possibilities_list if self.condition(grid, num, row, col)
        ]

    def print(self, grid: np.ndarray = None, refresh=0.2, override=False):
        if grid is None:
            grid = self.grid
        if (
            self.display_time is None
            or override
            or (datetime.now() - self.display_time) / timedelta(seconds=1) > refresh
        ):
            clear_output(wait=True)

            def print_divider():
                line_str = "-" * (4 * 2 + 2) + "+"
                line_str += "-" * (4 * 2 + 3) + "+"
                line_str += "-" * (4 * 2 + 2)
                print("+-" + line_str + "-+")

            print_divider()
            for row in range(9):
                line_str = bytearray("   ".join(grid[row, :].astype(str)), "utf-8")
                line_str[10] = bytearray("|", "utf-8")[0]
                line_str[22] = bytearray("|", "utf-8")[0]
                print("| " + line_str.decode() + " | ")
                if row in (2, 5, 8):
                    print_divider()
                elif row < 8:
                    print("| " + " " * 10 + "|" + " " * 11 + "|" + " " * 10 + " |")
            self.display_time = datetime.now()

    def deduce(self, grid: np.ndarray = None):
        if grid is None:
            grid = self.grid
        for row in range(9):
            if deduce(grid[row, :]):
                self.deduce(grid)
                return
        for col in range(9):
            if deduce(grid[:, col]):
                self.deduce(grid)
                return
        for block_row in range(0, 9, 3):
            for block_col in range(0, 9, 3):
                block = grid[block_row : block_row + 3, block_col : block_col + 3]
                if np.count_nonzero(block) == 8:
                    arr = block.flatten()
                    deduce(arr)
                    grid[
                        block_row : block_row + 3, block_col : block_col + 3
                    ] = arr.reshape([3, 3])
                    self.deduce(grid)
                    return

    def possibilities(self, grid: np.array, possibilities_list: list = None) -> list:
        """
        Generate list of possibilities. If possibilities_list is included, it will narrow
        down the previously known possibilities

        Parameters
        ----------
        grid: np.ndarray
            Grid representing puzzle, should be 9x9 array. Use 0 for unspecified entries.
        possibilities_list: list
            list of list of lists, innermost list is a list of numbers from 1 to 9,
            outer dimensions represent row and column. Includes a list of possibilities for
            every cell in the grid
        """
        if possibilities_list is None:
            return [
                [
                    None if grid[row, col] > 0 else self.candidates(grid, row, col)
                    for col in range(9)
                ]
                for row in range(9)
            ]
        else:
            return [
                [
                    None
                    if grid[row, col] > 0
                    else self.candidates(grid, row, col, possibilities_list[row][col])
                    for col in range(9)
                ]
                for row in range(9)
            ]

    def check_grid(self, grid: np.array = None) -> bool:
        if grid is None:
            grid = self.grid
        return all(
            (
                True
                if grid[row, col] == 0
                else self.condition(grid, grid[row, col], row, col)
                for row in range(9)
                for col in range(9)
            )
        )

    def count_errors(self, grid: np.array = None) -> int:
        if grid is None:
            grid = self.grid
        return sum(
            (
                not self.condition(grid, grid[row, col], row, col)
                for row in range(9)
                for col in range(9)
            )
        )

    def possible_cols(self, grid: np.array, num: int, row: int) -> list:
        return [
            col
            for col in range(9)
            if grid[row, col] == 0 and self.condition(grid, num, row, col)
        ]

    def _solve(
        self, grid: np.ndarray = None, verbose: bool = True, partial: bool = False
    ) -> np.ndarray:
        # Make sure we don't mutate the input
        if grid is None:
            grid = self.grid.copy()
        else:
            grid = grid.copy()
        self.deduce(grid)
        if verbose:
            self.print(grid)
        if not hasattr(self, "num_order"):
            self.num_order = get_num_order(grid)
        if partial and not hasattr(self, "partial_order"):
            self.partial_order = get_num_order(grid, partial=True)
        num_order = self.partial_order if partial else self.num_order
        for test_num in num_order:
            if np.count_nonzero(grid.copy() == test_num) < 9:
                for row in range(9):
                    if test_num in grid[row, :]:
                        continue
                    cols = self.possible_cols(grid, test_num, row)
                    if len(cols) == 0:
                        return
                    else:
                        shuffle(cols)
                        for col in cols:
                            grid_new = grid.copy()
                            grid_new[row, col] = test_num
                            result = self._solve(grid_new, verbose=verbose)
                            if result is not None:
                                return result
                            else:
                                continue
                        return
        if self.check_grid(grid):
            return grid

    def _solve_poss(
        self,
        grid: np.ndarray = None,
        possibilities_list: list = None,
        verbose: bool = True,
        depth: int = 1,
    ) -> np.ndarray:
        """Solve using squares with fewest options"""
        if grid is None:
            grid = self.grid
        self.deduce(grid)
        if depth > 5:
            return self._solve(grid)
        if verbose:
            self.print(grid)
        if not hasattr(self, "cell_order"):
            possibilities_list = self.possibilities(grid, possibilities_list)
            poss_num = np.array(
                [
                    [
                        999
                        if possibilities_list[row][col] is None
                        else len(possibilities_list[row][col])
                        for col in range(9)
                    ]
                    for row in range(9)
                ]
            )
            if poss_num.min() > 3:
                result = self._solve(grid)
                if verbose:
                    self.print(result, override=True)
                return result
            grid_coords = [(row, col) for row in range(9) for col in range(9)]
            self.cell_order = sorted(grid_coords, key=poss_num.__getitem__)
        for cell in self.cell_order:
            row, col = cell
            block_row = row // 3 * 3
            block_col = col // 3 * 3
            if grid[cell] > 0:
                continue
            candidates = set(range(1, 10))
            candidates -= set(grid[row, :])
            candidates -= set(grid[:, col])
            candidates -= set(
                grid[block_row : block_row + 3, block_col : block_col + 3].flatten()
            )
            candidates = list(candidates)
            if len(candidates) == 0:
                return
            if len(candidates) == 1:
                grid_new = grid.copy()
                grid_new[cell] = candidates.pop()
                return self._solve_poss(grid_new, depth=depth)
            shuffle(candidates)
            for candidate in candidates:
                grid_new = grid.copy()
                grid_new[cell] = candidate
                if self.condition(grid_new, candidate, row, col):
                    result = self._solve_poss(grid_new, depth=depth * len(candidates))
                    if result is not None:
                        return result

    def solve(self, grid: np.ndarray = None, verbose: bool = True) -> np.ndarray:
        """
        Recursively solve sudoku by filling the numbers used in the puzzle first, using
        backtracking and constraint programming
        
        Parameters
        ----------
        grid: np.ndarray, optional
            Grid representing puzzle, should be 9x9 array. Use 0 for unspecified entries.
            If unspecified, will use self.grid
        verbose: bool, default=True
            Whether to print grid while solving
        Returns
        -------
        np.ndarray
            Filled Grid
        """
        result = self._solve_poss(grid, verbose=verbose)
        if result is None:
            raise ValueError("No solution")
        else:
            if verbose:
                self.print(result, override=True)
            return result

    def diag(self, num: int, row: int, col: int):
        """
        Try to insert a number in the grid and print result of each test. Used for diagnostics.
        
        Parameters
        ----------
        num: int
            Number to test, between 1 and 9
        row: int
            Row to insert num in, between 0 and 8
        col: int
            Col to insert num in, between 0 and 8
        """
        for cond in self.condition.conditions:
            if isinstance(cond, type):
                print(cond.__name__, cond().test(self.grid, num, row, col))
            else:
                print(cond.__class__.__name__, cond.test(self.grid, num, row, col))


class StandardSolver(SudokuSolver):
    """Standard sudoku solver"""

    def __init__(self, grid: np.array):
        """
        Parameters
        ----------
        grid: np.ndarray
            Grid representing puzzle, should be 9x9 array. Use 0 for unspecified entries
        """
        # initialize using standard ruleset
        super().__init__(grid, ComboCondition(RookCondition, BlockCondition))


class CrypticSolver(SudokuSolver):
    """
    Cryptic sudoku solver, see 
    https://www.theguardian.com/science/2020/may/18/can-you-solve-it-sudoku-as-spectator-sport-is-unlikely-lockdown-hit
    """

    def __init__(self, grid: np.array):
        """
        Parameters
        ----------
        grid: np.ndarray
            Grid representing puzzle, should be 9x9 array. Use 0 for unspecified entries
        """
        # initialize using cryptic ruleset
        super().__init__(
            grid,
            ComboCondition(
                RookCondition,
                BlockCondition,
                KingCondition,
                KnightCondition,
                ConsecutiveCondition,
            ),
        )


def get_meat(arr, return_nums=True):
    one_pos = np.where(arr == 1)[0]
    nine_pos = np.where(arr == 9)[0]
    if len(one_pos) == 0 or len(nine_pos) == 0:
        return
    start = min(one_pos, nine_pos)[0] + 1
    stop = max(one_pos, nine_pos)[0]
    if return_nums:
        return start, stop, arr[start:stop]
    else:
        return start, stop


def get_meat_empty(arr, row_sum):
    one_pos = np.where(arr == 1)[0]
    nine_pos = np.where(arr == 9)[0]
    if len(one_pos) == 0 or len(nine_pos) == 0:
        return
    start = min(one_pos, nine_pos)[0] + 1
    stop = max(one_pos, nine_pos)[0]
    return (
        start,
        stop,
        row_sum - arr[start:stop].sum(),
        np.count_nonzero(arr[start:stop] == 0),
    )


def max_poss_meat(arr: np.array, minimum=False) -> int:
    """Return maximum possible sandwich sum"""
    start, stop, meat = get_meat(arr)
    missing_nos = [x for x in range(1, 10) if x not in arr]
    missing_nos = sorted(missing_nos, reverse=not minimum)
    zero_count = np.count_nonzero(meat == 0)
    max_sum = meat.sum() + sum(missing_nos[:zero_count])
    return max_sum


class SandwichCondition(SudokuCondition):
    def __init__(self, row_sums: np.array, col_sums: np.array):
        """
        Parameters
        ----------
        row_sums: np.ndarray
            array of sum of numbers sandwiched between 1 and 9 in each row
        col_sums: np.ndarray
            array of sum of numbers sandwiched between 1 and 9 in each column
        """
        self.row_sums = row_sums
        self.col_sums = col_sums

    def test(self, grid: np.ndarray, num: int, row: int, col: int) -> bool:
        #  unify column and row tests
        for dimension in ["row", "col"]:
            if dimension == "row":
                row_idx = row
                col_idx = col
                grid_iter = grid
                row_sum = self.row_sums[row]
            else:
                row_idx = col
                col_idx = row
                grid_iter = grid.T
                row_sum = self.col_sums[col]
            if row_sum is None:  # support for missing row sums
                continue
            temp_row = grid_iter[row_idx, :].copy()
            temp_row_nonzero = [x for x in temp_row if x != 0]
            if len(set(temp_row_nonzero)) != len(temp_row_nonzero):
                return False
            # make sure row sum is achievable if we're placing 1 or 9
            if num == 9 and 1 in temp_row:
                temp_row[col_idx] = 9
                if max_poss_meat(temp_row) < row_sum:
                    return False
                if max_poss_meat(temp_row, minimum=True) > row_sum:
                    return False
                # check for sufficient space
                dist = np.abs(col_idx - np.where(temp_row == 1)[0][0])
                if dist - 1 < SANDWICH_MIN_LENGTH[row_sum]:
                    return False
                if dist - 1 > SANDWICH_MAX_LENGTH[row_sum]:
                    return False
                continue
            temp_row = grid_iter[row_idx, :].copy()
            if num == 1 and 9 in temp_row:
                temp_row[col_idx] = 1
                if max_poss_meat(temp_row) < row_sum:
                    return False
                if max_poss_meat(temp_row, minimum=True) > row_sum:
                    return False
                # check for sufficient space
                dist = np.abs(col_idx - np.where(temp_row == 9)[0][0])
                if dist - 1 < SANDWICH_MIN_LENGTH[row_sum]:
                    return False
                if dist - 1 > SANDWICH_MAX_LENGTH[row_sum]:
                    return False
                continue
            if 1 not in grid_iter[row_idx, :] or 9 not in grid_iter[row_idx, :]:
                continue
            start, stop, _ = get_meat(grid_iter[row_idx, :])
            extent_checks = [
                (0, 1, 0),
                (1, 5, 1),
                (5, 9, 2),
                (9, 14, 3),
                (14, 20, 4),
                (20, 27, 5),
                (27, 35, 6),
            ]
            # check that 1 and 9 aren't too far apart
            for min_sum, max_sum, max_len in extent_checks:
                if row_sum >= min_sum and row_sum < max_sum and stop - start > max_len:
                    return False
            nums = grid_iter[row_idx, start:stop]
            if start == stop:  # 1 and 9 adjacent
                if row_sum > 0:
                    return False
                else:
                    continue
            elif nums.min() == 0:  # 1 and 9 present, holes between
                temp_row = grid_iter[row_idx, :]
                if max_poss_meat(temp_row) < row_sum:

                    return False
                if max_poss_meat(temp_row, minimum=True) > row_sum:

                    return False
                else:  # sum is okay thusfar
                    continue
            else:  # 1 and 9 present, no holes between
                if nums.sum() != row_sum:
                    return False
        return True


SANDWICH_COMBOS = dict()
for length in range(0, 8):
    for combo in combinations(range(2, 9), length):
        if (sum(combo), length) not in SANDWICH_COMBOS:
            SANDWICH_COMBOS[(sum(combo), length)] = []
        SANDWICH_COMBOS[(sum(combo), length)].append(set(combo))

SANDWICH_SEARCH_ORDER_DICT = {
    k: sum((len(list(permutations(x))) for x in v)) for k, v in SANDWICH_COMBOS.items()
}
SANDWICH_MIN_LENGTH = {
    rsum: min((v for k, v in SANDWICH_COMBOS.keys() if k == rsum))
    for rsum in set((k for k, _ in SANDWICH_COMBOS.keys()))
}
SANDWICH_MAX_LENGTH = {
    rsum: max((v for k, v in SANDWICH_COMBOS.keys() if k == rsum))
    for rsum in set((k for k, _ in SANDWICH_COMBOS.keys()))
}


class SandwichSolver(SudokuSolver):
    """
    Sandwich sudoku solver. See
    https://www.theguardian.com/science/2019/may/06/can-you-solve-it-sandwich-sudoku-a-new-puzzle-goes-viral
    """

    def __init__(self, grid: np.array, row_sums: np.array, col_sums: np.array):
        """
        Parameters
        ----------
        grid: np.ndarray
            Grid representing puzzle, should be 9x9 array. Use 0 for unspecified entries
        """
        # initialize using standard ruleset
        self.row_sums = row_sums
        self.col_sums = col_sums
        super().__init__(
            grid,
            ComboCondition(
                RookCondition, BlockCondition, SandwichCondition(row_sums, col_sums)
            ),
        )

    def possible_cols(self, grid: np.array, num: int, row: int) -> list:
        if 1 in grid[row, :] and 9 in grid[row, :]:
            if self.row_sums[row] in {2, 3, 4} and num == self.row_sums[row]:
                start, stop, _ = get_meat(grid[row, :])
                return list(range(start, stop))

        possibles = range(9)
        return [
            col
            for col in possibles
            if grid[row, col] == 0 and self.condition(grid, num, row, col)
        ]

    def print(self, grid: np.ndarray = None, refresh=0.05, override=False):
        if grid is None:
            grid = self.grid
        if (
            self.display_time is None
            or override
            or (datetime.now() - self.display_time) / timedelta(seconds=1) > refresh
        ):
            clear_output(wait=True)

            def print_divider():
                line_str = "-" * (4 * 2 + 2) + "+"
                line_str += "-" * (4 * 2 + 3) + "+"
                line_str += "-" * (4 * 2 + 2)
                print("+-" + line_str + "-+")

            print_divider()
            for row in range(9):
                line_str = bytearray("   ".join(grid[row, :].astype(str)), "utf-8")
                line_str[10] = bytearray("|", "utf-8")[0]
                line_str[22] = bytearray("|", "utf-8")[0]
                print(
                    "| "
                    + line_str.decode()
                    + " | "
                    + str(self.row_sums[row] if self.row_sums[row] is not None else "")
                )
                if row in (2, 5, 8):
                    print_divider()
                elif row < 8:
                    print("| " + " " * 10 + "|" + " " * 11 + "|" + " " * 10 + " |")
            col_str = ["    " if x is None else str(x).center(4) for x in self.col_sums]
            print(" " + "".join(col_str))
            self.display_time = datetime.now()

    def solve(self, grid: np.ndarray = None, verbose: bool = True) -> np.ndarray:
        """
        Recursively solve sudoku by filling the numbers used in the puzzle first, using
        backtracking and constraint programming
        
        Parameters
        ----------
        grid: np.ndarray, optional
            Grid representing puzzle, should be 9x9 array. Use 0 for unspecified entries.
            If unspecified, will use self.grid
        verbose: bool, default=True
            Whether to print grid while solving
        Returns
        -------
        np.ndarray
            Filled Grid
        """
        if grid is None:
            grid = self.grid
        self.deduce(grid)
        for row in range(9):
            if self.row_sums[row] == 35 and min(grid[row, 0], grid[row, -1]) == 0:
                for top, bot in [(1, 9), (9, 1)]:
                    grid_new = grid.copy()
                    if grid_new[row, 0] == 0:
                        grid_new[row, 0] = top
                    if grid_new[row, -1] == 0:
                        grid_new[row, -1] = bot
                    if grid_new[row, 0] == grid_new[row, -1]:
                        continue
                    try:
                        result = self.solve(grid_new, verbose=verbose)
                        return result
                    except ValueError:
                        continue
                raise ValueError("No solution")
        for col in range(9):
            if self.col_sums[col] == 35 and min(grid[0, col], grid[-1, col]) == 0:
                for top, bot in [(1, 9), (9, 1)]:
                    grid_new = grid.copy()
                    grid_new[0, col] = top
                    grid_new[-1, col] = bot
                    try:
                        result = self.solve(grid_new, verbose=verbose)
                        self.print(result)
                        return result
                    except ValueError:
                        continue
                raise ValueError("No solution")
        result = self._solve_sandwich(grid, verbose=verbose)
        if result is None:
            raise ValueError("No solution")
        else:
            if verbose:
                self.print(result, override=True)
            return result

    def _solve_sandwich(
        self,
        grid: np.ndarray = None,
        possibilities_list: list = None,
        verbose: bool = True,
    ) -> np.ndarray:
        # Make sure we don't mutate the input
        if grid is None:
            grid = self.grid.copy()
        else:
            grid = grid.copy()
        if verbose:
            self.print(grid)
        if not hasattr(self, "num_order"):
            self.num_order = get_num_order(grid)
            self.num_order = [1, 9] + [x for x in self.num_order if x not in {1, 9}]
        row_sums = self.row_sums.copy()
        col_sums = self.col_sums.copy()
        for test_num in self.num_order:
            if test_num in {1, 9}:
                if np.count_nonzero(grid.copy() == test_num) < 9:
                    for row in range(9):
                        if test_num in grid[row, :]:
                            continue
                        cols = self.possible_cols(grid, test_num, row)
                        if len(cols) == 0:
                            return
                        else:
                            for col in cols:
                                grid_new = grid.copy()
                                grid_new[row, col] = test_num
                                result = self._solve_sandwich(grid_new, verbose=verbose)
                                if result is not None:
                                    return result
                                else:
                                    continue
                            return
            else:  # filled all 1's and 9's
                if np.count_nonzero(grid) > 50:  # use standard solver
                    return self._solve_poss(grid, verbose=verbose)
                extents = {}
                for row in range(9):
                    if row_sums[row] is None:
                        continue
                    extents[row] = get_meat_empty(grid[row, :], row_sums[row])
                try:
                    search_order_dict = {}
                    for row, (start, stop, new_sum, slots) in extents.items():
                        if row_sums[row] is not None:
                            row_sums[row] = new_sum
                        search_order_dict[row] = SANDWICH_SEARCH_ORDER_DICT[
                            (new_sum, slots)
                        ]
                except KeyError:
                    return
                col_extents = {}
                for col in range(9):
                    if col_sums[col] is None:
                        continue
                    col_extents[col] = get_meat_empty(grid[:, col], col_sums[col])
                try:
                    col_search_order_dict = {}
                    for col, (start, stop, new_sum, slots) in col_extents.items():
                        if col_sums[col] is not None:
                            col_sums[col] = new_sum
                        col_search_order_dict[col] = SANDWICH_SEARCH_ORDER_DICT[
                            (new_sum, slots)
                        ]
                except KeyError:
                    return

                def check_sandwich_has_slots(arr):
                    start, stop, meat = get_meat(arr)
                    if start == stop:
                        return False
                    return meat.min() == 0

                search_order_dict = {
                    k: v
                    for k, v in search_order_dict.items()
                    if check_sandwich_has_slots(grid[k, :])
                }
                col_search_order_dict = {
                    k: v
                    for k, v in col_search_order_dict.items()
                    if check_sandwich_has_slots(grid[:, k])
                }
                search_order = sorted(
                    search_order_dict.keys(), key=search_order_dict.__getitem__
                )
                col_search_order = sorted(
                    col_search_order_dict.keys(), key=col_search_order_dict.__getitem__
                )
                # if the remaining searches are too deep, fall back to previous algorithm
                fallback = (
                    min(
                        min(col_search_order_dict.values())
                        if col_search_order_dict
                        else 99999,
                        min(search_order_dict.values()) if search_order_dict else 99999,
                    )
                    > 99999
                )
                if not col_search_order_dict and not search_order_dict:
                    fallback = True
                if fallback:
                    result = self._solve_poss(grid, verbose=verbose)
                    if result is not None:
                        return result
                    else:
                        return
                # row search is more efficient
                row_preferred = min(
                    list(col_search_order_dict.values()) + [99999]
                ) >= min(list(search_order_dict.values()) + [99999])
                if row_preferred:
                    for row in search_order:
                        if self.row_sums[row] is None:
                            continue
                        start, stop, row_sum, row_empty = extents[row]
                        if start == stop or grid[row, start:stop].min() > 0:
                            continue
                        if (row_sum, row_empty) not in SANDWICH_COMBOS:
                            return
                        combos = SANDWICH_COMBOS[(row_sum, row_empty)]
                        for combo in combos:
                            # for the given row, iterate over all valid combinations
                            # skip if there are invalid numbers in the combination
                            if combo.intersection(set(grid[row, start:stop])):
                                continue
                            for permutation in permutations(combo):
                                # iterate over all permutations
                                permutation = list(permutation)
                                grid_new = grid.copy()
                                for idx in range(start, stop):
                                    if grid_new[row, idx] == 0:
                                        grid_new[row, idx] = permutation.pop()
                                if not all(
                                    (
                                        self.condition(
                                            grid_new, grid_new[row, col], row, col
                                        )
                                        for col in range(start, stop)
                                    )
                                ):
                                    continue
                                result = self._solve_sandwich(
                                    grid_new,
                                    deepcopy(possibilities_list),
                                    verbose=verbose,
                                )
                                if result is not None:
                                    return result
                        return
                # col search is more efficient
                else:
                    for col in col_search_order:
                        if self.col_sums[col] is None:
                            pass
                        start, stop, col_sum, col_empty = col_extents[col]
                        if start == stop or grid[start:stop, col].min() > 0:
                            continue
                        if (col_sum, col_empty) not in SANDWICH_COMBOS:
                            return
                        combos = SANDWICH_COMBOS[(col_sum, col_empty)]
                        for combo in combos:
                            # for the given row, iterate over all valid combinations
                            # skip if there are invalid numbers in the combination
                            if combo.intersection(set(grid[start:stop, col])):
                                continue
                            for permutation in permutations(combo):
                                # iterate over all permutations
                                permutation = list(permutation)
                                grid_new = grid.copy()
                                for idx in range(start, stop):
                                    if grid_new[idx, col] == 0:
                                        grid_new[idx, col] = permutation.pop()
                                if not all(
                                    (
                                        self.condition(
                                            grid_new, grid_new[row, col], row, col
                                        )
                                        for row in range(start, stop)
                                    )
                                ):
                                    continue
                                result = self._solve_sandwich(
                                    grid_new,
                                    deepcopy(possibilities_list),
                                    verbose=verbose,
                                )
                                if result is not None:
                                    return result
                        return
        if self.check_grid(grid) and grid.min() > 0:
            return grid


class ThermometerCondition(SudokuCondition):
    def __init__(self, paths: list):
        """
        Parameters
        ----------
        paths: list of list of tuples
            List of list of coordinates that should have increasing numbers
        """
        self.paths = sorted(paths, key=len, reverse=True)

    def test(self, grid: np.ndarray, num: int, row: int, col: int) -> bool:
        grid = grid.copy()
        grid[row, col] = num
        for path in self.paths:
            if (row, col) not in path:
                pass
            nums = [grid[row, col] for row, col in path if grid[row, col] > 0]
            if not np.all(np.diff(nums) > 0):
                return False
        # now check for numbers that are too close together
        for path in self.paths:
            if (row, col) not in path:
                pass
            position_dict = {
                grid[row, col]: idx
                for idx, (row, col) in enumerate(path)
                if grid[row, col] != 0
            }
            # check for numbers too small near beginning of too large near end
            for test_num, idx in position_dict.items():
                if test_num < idx + 1 or 9 - test_num < len(path) - idx - 1:
                    return False
            if len(position_dict) < 2:
                continue
            position_keys = sorted(position_dict.keys())
            # check for insufficient space between numbers
            for idx in range(len(position_keys) - 1):
                p, q = position_keys[idx], position_keys[idx + 1]
                if position_dict[q] - position_dict[p] > q - p:
                    return False
        return True


def _sudoku_update(grid, possibilities_list, row, col, paths=None):
    """incremental possibilities update"""
    num = grid[row, col]
    for row2 in range(9):
        for col2 in range(9):
            if possibilities_list[row2][col2] is None:  # cell already filled
                continue
            if row == row2 and col == col2:  # we just updated this
                possibilities_list[row][col] = None
            elif (
                col2 == col
                or row2 == row
                or (col2 // 3 == col // 3 and row2 // 3 == row2 // 3)
            ):  # same row or column
                possibilities_list[row2][col2] = [
                    x for x in possibilities_list[row2][col2] if x != num
                ]
    # path check
    if paths is None:
        return
    for path in paths:
        if (row, col) not in path:  # not a relevant path
            continue
        for row2, col2 in path:
            if row == row2 and col == col2:  # same cell, continue
                continue
            if possibilities_list[row2][col2] is None:  # already filled, continue
                continue
            else:  # remove num from this cell's possibility list
                possibilities_list[row2][col2] = [
                    x for x in possibilities_list[row2][col2] if x != num
                ]


class ThermometerSolver(SudokuSolver):
    """
    Thermometer sudoku solver. See
    https://www.gmpuzzles.com/blog/sudoku-rules-and-info/thermo-sudoku-rules-and-info/
    """

    def __init__(self, grid: np.array, paths: list):
        """
        Parameters
        ----------
        grid: np.ndarray
            Grid representing puzzle, should be 9x9 array. Use 0 for unspecified entries
        paths: list of list of tuples
            List of list of coordinates that should have increasing numbers
        """
        # initialize using standard ruleset
        def comb(x):
            length = len(x)
            return min(length, 9 - length)

        self.paths = sorted(paths, key=comb)
        super().__init__(
            grid,
            ComboCondition(RookCondition, BlockCondition, ThermometerCondition(paths)),
        )

    def solve(self, grid: np.ndarray = None, verbose: bool = True) -> np.ndarray:
        """
        Recursively solve sudoku by filling the numbers used in the puzzle first, using
        backtracking and constraint programming
        
        Parameters
        ----------
        grid: np.ndarray, optional
            Grid representing puzzle, should be 9x9 array. Use 0 for unspecified entries.
            If unspecified, will use self.grid
        verbose: bool, default=True
            Whether to print grid while solving
        Returns
        -------
        np.ndarray
            Filled Grid
        """
        result = self._solve_thermometer(grid, verbose=verbose)
        if result is None:
            raise ValueError("No solution")
        else:
            return result

    def _solve_thermometer(
        self, grid: np.ndarray = None, verbose: bool = True, possibilities_list=None
    ) -> np.ndarray:
        # Make sure we don't mutate the input
        if grid is None:
            grid = self.grid.copy()
        else:
            grid = grid.copy()
        self.deduce(grid)
        if verbose:
            self.print(grid)
        possibilities_list = self.possibilities(grid, possibilities_list)
        if min([len(x) for x in self.possibilities(grid)]) == 0:
            return False
        # check for any cells we can deduce or contradictions
        for row in range(9):
            for col in range(9):
                if grid[row, col] > 0:
                    continue
                if len(possibilities_list[row][col]) == 1:
                    grid_new = grid.copy()
                    grid_new[row, col] = possibilities_list[row][col][0]
                    result = self._solve_thermometer(grid_new, possibilities_list)
                    if result is not None:
                        return result
                    else:
                        return
                elif len(possibilities_list[row][col]) == 0:
                    return
        # if we don't have much filled yet, do some branching if there are only 2 possibilities
        if np.count_nonzero(grid) <= 15:
            for row in range(9):
                for col in range(9):
                    if grid[row, col] > 0:
                        continue
                    candidates = possibilities_list[row][col]
                    if len(candidates) == 1:
                        grid[row, col] = candidates[0]
                        possibilities_list[row][col] = None
                    elif len(candidates) <= 2 and len(candidates) > 0:
                        for num in candidates:
                            grid_new = grid.copy()
                            grid_new[row, col] = num
                            result = self._solve_thermometer(
                                grid_new, deepcopy(possibilities_list)
                            )
                            if result is not None:
                                return result
                        return
        # fill paths first
        for path in self.paths:
            path_nums = [grid[row, col] for row, col in path]
            if min(path_nums) > 0:  # path is filled
                continue

            for segments in split_path(np.array(path_nums)):
                try:
                    for seg_slice, seg_nums in segments:
                        grid_new = grid.copy()
                        for cell, num in zip(path[seg_slice], seg_nums):
                            grid_new[cell] = num
                        for cell, num in zip(path[seg_slice], seg_nums):
                            if not self.condition(grid_new, num, cell[0], cell[1]):
                                raise AssertionError
                except AssertionError:
                    continue
                result = self._solve_thermometer(grid_new)
                if result is not None:
                    return result
                else:
                    continue
                return
            return
        # use std sudoku solver for speed because we already fulfilled thermometer conds
        result = StandardSolver(grid)._solve_poss(verbose=verbose)
        if result is not None:
            return result
        else:
            return
