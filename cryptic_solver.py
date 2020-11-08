#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import typing
from copy import deepcopy
import matplotlib.pyplot as plt
import time
from itertools import combinations, permutations
from random import shuffle


def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")


if is_interactive():
    from IPython.display import clear_output
else:
    # don't bother without put clearing if we're not in IPython
    def clear_output(*args, **kwargs):
        return


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

    def __init__(self, conditions: SudokuCondition):
        self.conditions = conditions

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

    def candidates(
        self, grid: np.array, row: int, col: int, possibilities_list: list = None
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

    def check_grid(self, grid: np.array) -> bool:
        return all(
            (
                True
                if grid[row, col] == 0
                else self.condition(grid, grid[row, col], row, col)
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

    def _solve(self, grid: np.ndarray = None, verbose: bool = True) -> np.ndarray:
        # Make sure we don't mutate the input
        if grid is None:
            grid = self.grid.copy()
        else:
            grid = grid.copy()
        if verbose:
            clear_output(wait=True)
            print(grid)
        nums, counts = np.unique(grid, return_counts=True)
        num_order = sorted(range(len(nums)), key=counts.__getitem__, reverse=True)
        num_order = [nums[x] for x in num_order]
        num_order = num_order + [x for x in range(10) if x not in num_order]
        num_order = [x for x in num_order if x != 0]
        for test_num in num_order:
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
                            result = self._solve(grid_new, verbose=verbose)
                            if result is not None:
                                return result
                            else:
                                continue
                        return
        if self.check_grid(grid):
            return grid

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
        result = self._solve(grid, verbose=verbose)
        if result is None:
            raise ValueError("No solution")
        else:
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
        for cond in [
            RookCondition,
            BlockCondition,
            KingCondition,
            KnightCondition,
            ConsecutiveCondition,
        ]:
            print(cond.__name__, cond().test(self.grid, num, row, col))


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
        super().__init__(grid, ComboCondition([RookCondition, BlockCondition]))


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
                [
                    RookCondition,
                    BlockCondition,
                    KingCondition,
                    KnightCondition,
                    ConsecutiveCondition,
                ]
            ),
        )


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
        for grid_iter, rc_sum in zip([grid, grid.T], [self.row_sums, self.col_sums]):
            for row, row_sum in enumerate(rc_sum):
                one_pos = np.where(grid_iter[row, :] == 1)[0]
                nine_pos = np.where(grid_iter[row, :] == 9)[0]
                if len(one_pos) == 0 or len(nine_pos) == 0:
                    continue
                start = min(one_pos, nine_pos)[0] + 1
                stop = max(one_pos, nine_pos)[0]
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
                    if (
                        row_sum >= min_sum
                        and row_sum < max_sum
                        and stop - start > max_len
                    ):
                        return False
                nums = grid_iter[row, start:stop]
                if start == stop:  # 1 and 9 adjacent
                    if row_sum > 0:
                        return False
                    else:
                        continue
                elif nums.min() == 0:  # 1 and 9 present, holes between
                    if nums.sum() > row_sum:  # sum is to high
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


def get_meat(arr):
    one_pos = np.where(arr == 1)[0]
    nine_pos = np.where(arr == 9)[0]
    if len(one_pos) == 0 or len(nine_pos) == 0:
        return
    start = min(one_pos, nine_pos)[0] + 1
    stop = max(one_pos, nine_pos)[0]
    if start == stop:
        return
    return start, stop, arr[start:stop]


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
                [RookCondition, BlockCondition, SandwichCondition(row_sums, col_sums)]
            ),
        )

    def possible_cols(self, grid: np.array, num: int, row: int) -> list:
        if 1 in grid[row, :] and 9 in grid[row, :]:
            if self.row_sums[row] in {2, 3, 4} and num == self.row_sums[row]:
                one_pos = np.where(grid[row, :] == 1)[0]
                nine_pos = np.where(grid[row, :] == 9)[0]
                start = min(one_pos, nine_pos)[0] + 1
                stop = max(one_pos, nine_pos)[0]
                return list(range(start, stop))

        possibles = range(9)
        return [
            col
            for col in possibles
            if grid[row, col] == 0 and self.condition(grid, num, row, col)
        ]

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
        result = self._solve_sandwich(grid, verbose=verbose)
        if result is None:
            raise ValueError("No solution")
        else:
            return result

    def _solve_sandwich(
        self, grid: np.ndarray = None, verbose: bool = True
    ) -> np.ndarray:
        # Make sure we don't mutate the input
        if grid is None:
            grid = self.grid.copy()
        else:
            grid = grid.copy()
        if verbose:
            clear_output(wait=True)
            print(grid)
        nums, counts = np.unique(grid, return_counts=True)
        num_order = sorted(range(len(nums)), key=counts.__getitem__, reverse=True)
        num_order = [nums[x] for x in num_order]
        num_order = num_order + [x for x in range(10) if x not in num_order]
        num_order = [x for x in num_order if x != 0]
        num_order = [1, 9] + [x for x in num_order if x not in {1, 9}]
        for test_num in num_order:
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
            else:
                if np.count_nonzero(grid) > 60:
                    return self.solve(grid, verbose=verbose)
                extents = []
                for row in range(9):
                    one_pos = np.where(grid[row, :] == 1)[0]
                    nine_pos = np.where(grid[row, :] == 9)[0]
                    start = min(one_pos, nine_pos)[0] + 1
                    stop = max(one_pos, nine_pos)[0]
                    extents.append((start, stop))
                try:
                    search_order_dict = {
                        row: SANDWICH_SEARCH_ORDER_DICT[
                            (self.row_sums[row], stop - start)
                        ]
                        for row, (start, stop) in enumerate(extents)
                    }
                except KeyError:
                    return
                col_extents = []
                for col in range(9):
                    one_pos = np.where(grid[:, col] == 1)[0]
                    nine_pos = np.where(grid[:, col] == 9)[0]
                    start = min(one_pos, nine_pos)[0] + 1
                    stop = max(one_pos, nine_pos)[0]
                    col_extents.append((start, stop))
                try:
                    col_search_order_dict = {
                        col: SANDWICH_SEARCH_ORDER_DICT[
                            (self.col_sums[col], stop - start)
                        ]
                        for col, (start, stop) in enumerate(col_extents)
                    }
                except KeyError:
                    return
                search_order_dict = {
                    k: v
                    for k, v in search_order_dict.items()
                    if get_meat(grid[k, :]) is not None
                    and get_meat(grid[k, :])[2].min() == 0
                }
                col_search_order_dict = {
                    k: v
                    for k, v in col_search_order_dict.items()
                    if get_meat(grid[:, k]) is not None
                    and get_meat(grid[:, k])[2].min() == 0
                }
                search_order = sorted(
                    search_order_dict.keys(), key=search_order_dict.__getitem__
                )
                col_search_order = sorted(
                    col_search_order_dict.keys(), key=col_search_order_dict.__getitem__
                )
                if (
                    min(
                        min(col_search_order_dict.values()),
                        min(search_order_dict.values()),
                    )
                    > 200
                ):
                    result = self._solve(grid, verbose=verbose)
                    if result is not None:
                        return result
                    else:
                        return
                elif min(col_search_order_dict.values()) >= min(
                    search_order_dict.values()
                ):
                    for row in search_order:
                        start, stop = extents[row]
                        if start == stop or grid[row, start:stop].min() > 0:
                            continue
                        if (self.row_sums[row], stop - start) not in SANDWICH_COMBOS:
                            return
                        combos = SANDWICH_COMBOS[(self.row_sums[row], stop - start)]
                        for combo in combos:
                            if set(grid[row, start:stop]) - combo - {0}:
                                continue
                            for permutation in permutations(combo):
                                if any(
                                    (grid[row, start:stop] != np.array(permutation))
                                    & (grid[row, start:stop] != 0)
                                ):
                                    continue
                                grid_new = grid.copy()
                                grid_new[row, start:stop] = permutation
                                result = self._solve_sandwich(grid_new, verbose=verbose)
                                if result is not None:
                                    return result
                        return
                else:
                    for col in col_search_order:
                        start, stop = col_extents[col]
                        if start == stop or grid[start:stop, col].min() > 0:
                            continue
                        if (self.col_sums[col], stop - start) not in SANDWICH_COMBOS:
                            return
                        combos = SANDWICH_COMBOS[(self.col_sums[col], stop - start)]
                        for combo in combos:
                            if set(grid[start:stop, col]) - combo - {0}:
                                continue
                            for permutation in permutations(combo):
                                if any(
                                    (grid[start:stop, col] != np.array(permutation))
                                    & (grid[start:stop, col] != 0)
                                ):
                                    continue
                                grid_new = grid.copy()
                                grid_new[start:stop, col] = permutation
                                if not self.check_grid(grid_new):
                                    continue
                                result = self._solve_sandwich(grid_new, verbose=verbose)
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
            nums = [grid[row, col] for row, col in path if grid[row, col] > 0]
            if not np.all(np.diff(nums) > 0):
                return False
        # now check for numberse that are too close together
        for path in self.paths:
            position_dict = {
                grid[row, col]: idx
                for idx, (row, col) in enumerate(path)
                if grid[row, col] != 0
            }
            for test_num, idx in position_dict.items():
                if test_num < idx + 1 or 9 - test_num < len(path) - idx - 1:
                    return False
            if len(position_dict) < 2:
                continue
            position_keys = sorted(position_dict.keys())
            for idx in range(len(position_keys) - 1):
                p, q = position_keys[idx], position_keys[idx + 1]
                if position_dict[q] - position_dict[p] > q - p:
                    return False
        return True


def sudoku_update(grid, possibilities_list, row, col, paths=[]):
    num = grid[row, col]
    for row2 in range(9):
        for col2 in range(9):
            if possibilities_list[row2][col2] is None:
                continue
            if row == row2 and col == col2:
                possibilities_list[row][col] = None
            elif col2 == col or row2 == row:
                possibilities_list[row2][col2] = [
                    x for x in possibilities_list[row2][col2] if x != num
                ]
    for path in paths:
        if (row, col) not in path:
            continue
        for row2, col2 in path:
            if row == row2 and col == col2:
                continue
            if possibilities_list[row2][col2] is None:
                continue
            else:
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
            return min(length, 9-length)

        self.paths = sorted(paths, key=comb)
        super().__init__(
            grid,
            ComboCondition(
                [RookCondition, BlockCondition, ThermometerCondition(paths)]
            ),
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
        if verbose:
            clear_output(wait=True)
            print(grid)
        possibilities_list = self.possibilities(grid, possibilities_list)
        if min([len(x) for x in self.possibilities(grid)]) == 0:
            return False
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
        if np.count_nonzero(grid) <= 15:
            for row in range(9):
                for col in range(9):
                    if grid[row, col] > 0:
                        continue
                    p_list = possibilities_list[row][col]
                    if len(p_list) == 1:
                        grid[row, col] = p_list[0]
                        p_list[row][col] = None
                    elif len(p_list) <= 2 and len(p_list) > 0:
                        for num in p_list:
                            grid_new = grid.copy()
                            grid_new[row, col] = num
                            result = self._solve_thermometer(
                                grid_new, deepcopy(possibilities_list)
                            )
                            if result is not None:
                                return result

        for path in self.paths:
            path_nums = [grid[row, col] for row, col in path]
            if min(path_nums) > 0:
                continue
            for row, col in path:
                if grid[row, col] > 0:
                    continue
                candidates = possibilities_list[row][col]
                if len(candidates) == 0:
                    return
                for num in candidates:
                    grid_new = grid.copy()
                    grid_new[row, col] = num
                    possibilities_list_new = deepcopy(possibilities_list)
                    sudoku_update(grid, possibilities_list_new, row, col, self.paths)
                    try:
                        for row2 in range(9):
                            for col2 in range(9):
                                cand = possibilities_list_new[row2][col2]
                                if cand is None:
                                    continue
                                if len(cand) == 1:
                                    grid[row2, col2] == cand[0]
                                    sudoku_update(
                                        grid,
                                        possibilities_list_new,
                                        row2,
                                        col2,
                                        self.paths,
                                    )
                                elif len(cand) == 0:
                                    raise AssertionError
                    except AssertionError:
                        continue
                    result = self._solve_thermometer(
                        grid_new,
                        verbose=verbose,
                        possibilities_list=possibilities_list_new,
                    )
                    if result is not None:
                        return result
                    else:
                        continue
                return
        for row2 in range(9):
            for col2 in range(9):
                cand = self.candidates(grid, row2, col2)
                if len(cand) == 1:
                    grid[row2, col2] == cand[0]
                elif len(cand) == 0:
                    return
        if not self.check_grid(grid):
            return
        result = StandardSolver(grid)._solve(verbose=verbose)
        if result is not None:
            return result
        else:
            return