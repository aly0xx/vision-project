# Step 6a: checks a sudoku grid for repeated digits and solves it with backtracking,
# always filling the empty cell with the fewest options first. Can also count solutions (a real puzzle has one).

import numpy as np


def options(grid, r, c):
	used = set(grid[r, :]) | set(grid[:, c])
	br, bc = r // 3 * 3, c // 3 * 3
	used |= set(grid[br:br + 3, bc:bc + 3].ravel())
	return [d for d in range(1, 10) if d not in used]


def is_valid(grid):
	for i in range(9):
		box = grid[i // 3 * 3:i // 3 * 3 + 3, i % 3 * 3:i % 3 * 3 + 3].ravel()
		for group in (grid[i, :], grid[:, i], box):
			digits = group[group > 0]
			if len(digits) != len(set(digits)):
				return False
	return True


def solve(grid):
	grid = np.array(grid, dtype=int)
	if not is_valid(grid):
		return None
	if backtrack(grid):
		return grid
	return None


def most_constrained(grid):
	best = None
	for r in range(9):
		for c in range(9):
			if grid[r, c] == 0:
				opts = options(grid, r, c)
				if best is None or len(opts) < len(best[2]):
					best = (r, c, opts)
	return best


def backtrack(grid):
	best = most_constrained(grid)
	if best is None:
		return True
	r, c, opts = best
	for d in opts:
		grid[r, c] = d
		if backtrack(grid):
			return True
	grid[r, c] = 0
	return False


def count_solutions(grid, limit=2):
	grid = np.array(grid, dtype=int)
	if not is_valid(grid):
		return 0
	return count(grid, limit)


def count(grid, limit):
	best = most_constrained(grid)
	if best is None:
		return 1
	r, c, opts = best
	total = 0
	for d in opts:
		grid[r, c] = d
		total += count(grid, limit - total)
		if total >= limit:
			break
	grid[r, c] = 0
	return total


if __name__ == "__main__":
	from utils import load_ground_truth

	puzzle = np.array(load_ground_truth("puzzle01"))
	solved = solve(puzzle)
	print(solved)
	print("valid:", is_valid(solved), "| givens kept:", bool(np.all(solved[puzzle > 0] == puzzle[puzzle > 0])))
	print("number of solutions:", count_solutions(puzzle))
