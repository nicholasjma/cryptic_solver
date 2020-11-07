#!/usr/bin/env python
# coding: utf-8

# In[211]:


import numpy as np
import typing
from copy import deepcopy
import matplotlib.pyplot as plt


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
        """Syntactic sugar to treat class isntance as function, see `test()`"""
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
    """Make sure the same number does not exist in the same 3x3 block"""

    def test(self, grid: np.ndarray, num: int, row: int, col: int) -> bool:
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
                assert condition().test(grid, num, row, col)
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

    def solve(
        self, grid: np.ndarray = None, possibilities_list: list = None
    ) -> np.ndarray:
        """
        Recursively solve sudoku using backtracking and constraint programming
        
        Parameters
        ----------
        grid: np.ndarray, optional
            Grid representing puzzle, should be 9x9 array. Use 0 for unspecified entries.
            If unspecified, will use self.grid
        possibilities list: list
            List of possibilities, should be list of lists of lists, see possibilities()
        """
        # Make sure we don't mutate the input
        if grid is None:
            grid = self.grid.copy()
        else:
            grid = grid.copy()
        # Compute all possibilities, using what we already know
        possibilities_list = self.possibilities(grid, possibilities_list)
        # Compute number of possiblities for each cell. 99 means we already filled that cell
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
        # Choose a cell to try that has the least number of possibilities
        search_x, search_y = np.unravel_index(np.argmin(poss_num), poss_num.shape)
        # Iterate over all possibilities for our search posdition
        for num in possibilities_list[search_x][search_y]:
            grid[search_x, search_y] = num
            # Clear previous output and print the grid
            clear_output(wait=True)
            print(grid)
            # Recompute possibilities
            poss_check = self.possibilities(grid, possibilities_list)
            poss_check_num = np.array(
                [
                    [
                        99
                        if poss_check[row][col] is None
                        else len(poss_check[row][col])
                        for col in range(9)
                    ]
                    for row in range(9)
                ]
            )
            if poss_check_num.min() == 0:  # we have an unfillable cell
                continue  # try the next number
            elif grid.min() == 1:  # we succeeded!
                self.grid = grid
                return grid
            else:  # so far so good, recursively try another cell
                result = self.solve(grid, poss_check)
                if (
                    result is not None
                ):  # if a solution is returned, keep return it recursively
                    self.grid = result
                    return result
        return

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


# In[212]:


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

