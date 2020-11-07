#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import typing
from copy import deepcopy
import matplotlib.pyplot as plt
import time


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


class ThermometerCondition(SudokuCondition):
    def __init__(self, paths: list):
        """
        Parameters
        ----------
        paths: list of list of tuples
            List of list of coordinates that should have increasing numbers
        """
        self.paths = paths

    def test(self, grid: np.ndarray, num: int, row: int, col: int) -> bool:
        for path in self.paths:
            nums = [grid[row, col] for row, col in path if grid[row, col] > 0]
            if not np.all(np.diff(nums) > 0):
                return False
        return True


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
                continue
                one_pos = np.where(grid_iter[row, :] == 1)[0]
                nine_pos = np.where(grid_iter[row, :] == 9)[0]
                if len(one_pos) == 0 or len(nine_pos) == 0:
                    print("missing 1 or 9")
                    continue
                else:
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
                            and start - stop > max_len
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
            num
            for num in possibilities_list
            if self.condition.test(grid, num, row, col)
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

    def _solve(self, grid: np.ndarray = None) -> np.ndarray:
        # Make sure we don't mutate the input
        if grid is None:
            grid = self.grid.copy()
        else:
            grid = grid.copy()
        clear_output(wait=True)
        print(grid)
        nums, counts = np.unique(grid, return_counts=True)
        if not hasattr(self, "num_order"):
            num_order = sorted(range(len(nums)), key=counts.__getitem__, reverse=True)
            num_order = [nums[x] for x in num_order]
            num_order = num_order + [x for x in range(10) if x not in num_order]
            num_order = [x for x in num_order if x != 0]
            self.num_order = num_order
        for test_num in self.num_order:
            if np.count_nonzero(grid.copy() == test_num) < 9:
                for row in range(10):
                    if test_num in grid[row, :]:
                        continue
                    cols = self.possible_cols(grid, test_num, row)
                    if len(cols) == 0:
                        return
                    else:
                        for col in cols:
                            grid_new = grid.copy()
                            grid_new[row, col] = test_num
                            result = self._solve(grid_new)
                            if result is not None:
                                return result
                            else:
                                continue
                        return
        if self.check_grid(grid):
            return grid

    def solve(self, grid: np.ndarray = None) -> np.ndarray:
        """
        Recursively solve sudoku by filling the numbers used in the puzzle first, using
        backtracking and constraint programming
        
        Parameters
        ----------
        grid: np.ndarray, optional
            Grid representing puzzle, should be 9x9 array. Use 0 for unspecified entries.
            If unspecified, will use self.grid
        possibilities list: list
            List of possibilities, should be list of lists of lists, see possibilities()
            
        Returns
        -------
        np.ndarray
            Filled Grid
        """
        result = self._solve(grid)
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


class StandardSudokuSolver(SudokuSolver):
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
        super().__init__(
            grid,
            ComboCondition(
                [RookCondition, BlockCondition, SandwichCondition(row_sums, col_sums)]
            ),
        )

    def possible_cols(self, grid: np.array, num: int, row: int) -> list:
        return [
            col
            for col in range(9)
            if grid[row, col] == 0 and self.condition(grid, num, row, col)
        ]

    def _solve(self, grid: np.ndarray = None) -> np.ndarray:
        # Make sure we don't mutate the input
        if grid is None:
            grid = self.grid.copy()
        else:
            grid = grid.copy()
        clear_output(wait=True)
        print(grid)
        nums, counts = np.unique(grid, return_counts=True)
        if not hasattr(self, "num_order"):
            num_order = sorted(range(len(nums)), key=counts.__getitem__, reverse=True)
            num_order = [nums[x] for x in num_order]
            num_order = num_order + [x for x in range(10) if x not in num_order]
            num_order = [x for x in num_order if x != 0]
            self.num_order = [x for x in num_order if x in {1, 9}] + [
                x for x in num_order if x not in {1, 9}
            ]
        for test_num in self.num_order:
            if np.count_nonzero(grid.copy() == test_num) < 9:
                for row in range(10):
                    if test_num in grid[row, :]:
                        continue
                    cols = self.possible_cols(grid, test_num, row)
                    if len(cols) == 0:
                        return
                    else:
                        for col in cols:
                            grid_new = grid.copy()
                            grid_new[row, col] = test_num
                            result = self._solve(grid_new)
                            if result is not None:
                                return result
                            else:
                                continue
                        return
        if self.check_grid(grid):
            return grid


# In[2]:


cs = CrypticSolver(
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
)

cs.solve()


# In[3]:


cs = SandwichSolver(
    np.array(
        [
            [6, 1, 2, 9, 3, 8, 4, 7, 5],
            [7, 8, 3, 4, 5, 1, 2, 6, 9],
            [0, 0, 9, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 6, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 7, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
        ]
    ),
    row_sums=np.array([2, 8, 26, 29, 0, 23, 15, 2, 4]),
    col_sums=np.array([10, 23, 23, 23, 14, 12, 21, 0, 0]),
)

cs.solve()

