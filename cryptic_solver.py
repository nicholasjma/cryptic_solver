#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import typing
from copy import deepcopy
import matplotlib.pyplot as plt


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

if is_interactive():
    from IPython.display import clear_output
else:
    # don't bother without put clearing if we're not in IPython
    clear_output = lambda: None


class SudokuCondition:
    """Template for sudoku condition"""

    def test(self, grid: np.ndarray, num: int, row: int, col: int) -> bool:
        """Implement testing here, return True if valid"""
        pass

    def __call__(self, grid: np.ndarray, num: int, row: int, col: int) -> bool:
        return self.test(grid, num, row, col)


class RookCondition(SudokuCondition):
    """Make sure the same number does not exist in the same row or col"""

    def test(self, grid: np.ndarray, num: int, row: int, col: int) -> bool:
        try:
            assert num not in grid[row, :]
            assert num not in grid[:, col]
        except AssertionError:
            return False
        else:
            return True


class BlockCondition(SudokuCondition):
    def test(self, grid: np.ndarray, num: int, row: int, col: int) -> bool:
        block_row = (row // 3) * 3
        block_col = (col // 3) * 3
        return num not in grid[block_row : block_row + 3, block_col : block_col + 3]


class KingCondition(SudokuCondition):
    """Make sure diagonals are not the same"""

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
    def __init__(self, conditions):
        self.conditions = conditions

    def test(self, grid: np.ndarray, num: int, row: int, col: int) -> bool:
        try:
            for condition in self.conditions:
                assert condition().test(grid, num, row, col)
        except AssertionError:
            return False
        else:
            return True


class SudokuSolver:
    def __init__(self, grid: np.array, condition):
        self.grid = grid
        self.condition = condition

    def candidates(self, grid, row, col, possibilities_list: list = None):
        if possibilities_list is None:
            possibilities_list = range(1, 10)
        return [
            num
            for num in possibilities_list
            if self.condition.test(grid, num, row, col)
        ]

    def possibilities(self, grid, possibilities_list: list = None) -> list:
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


class StandardSudokuSolver(SudokuSolver):
    def __init__(self, grid: np.array):
        super().__init__(grid, ComboCondition([RookCondition, BlockCondition]))


class CrypticSolver(SudokuSolver):
    def __init__(self, grid: np.array):
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

    def solve(self, grid=None, possibilities_list=None):
        """
        Recursively solve sudoku using 
        """
        if grid is None:
            grid = self.grid.copy()
        else:
            grid = grid.copy()
        clear_output(wait=True)
        print(grid)
        possibilities_list = self.possibilities(grid, possibilities_list)

        poss_num = np.array(
            [
                [
                    99
                    if possibilities_list[row][col] is None
                    else len(possibilities_list[row][col])
                    for col in range(9)
                ]
                for row in range(9)
            ]
        )
        search_x, search_y = np.unravel_index(np.argmin(poss_num), poss_num.shape)
        for num in possibilities_list[search_x][search_y]:
            grid[search_x, search_y] = num
            poss_check = self.possibilities(grid, possibilities_list)
            poss_check_num = np.array(
                [
                    [
                        999
                        if poss_check[row][col] is None
                        else len(poss_check[row][col])
                        for col in range(9)
                    ]
                    for row in range(9)
                ]
            )
            poss_check_min = poss_check_num.min()
            if poss_check_min == 0:
                continue
            elif grid.min() == 1:
                self.grid = grid
                return grid
            else:
                result = self.solve(grid, poss_check)
                if result is not None:
                    self.grid = result
                    return result
        return

    def diag(self, num, row, col):
        """
        Try to insert a number in the grid and print result of each test. Used for diagnostics.
        """
        for cond in [
            RookCondition,
            BlockCondition,
            KingCondition,
            KnightCondition,
            ConsecutiveCondition,
        ]:
            print(cond.__name__, cond().test(self.grid, num, row, col))


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

