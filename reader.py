# Full pipeline: finds the grid, picks the orientation the classifier is most sure about, reads the digits,
# solves the puzzle and draws the missing digits back onto the original photo with the inverse homography.

import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

from cells import ORIENTATIONS, binarize_grid, orient, process_cells, unorient
from classify import MODELS_PATH
from grid import find_grid
from solver import count_solutions, solve
from utils import RESULTS_DIR, load_photo, show_images
from warp import split_cells, warp_grid


def load_model(name=None):
	with open(MODELS_PATH, "rb") as f:
		saved = pickle.load(f)
	return saved["models"][name or saved["best"]]


def read_cells(binary, model):
	digits = process_cells(split_cells(binary))
	filled = [i for i, d in enumerate(digits) if d is not None]
	probs = np.zeros((81, 9))
	if filled:
		X = np.array([digits[i].ravel() / 255.0 for i in filled])
		probs[filled] = model.predict_proba(X)
	grid = np.zeros(81, dtype=int)
	grid[filled] = probs[filled].argmax(axis=1) + 1
	confidence = probs[filled].max(axis=1).mean() if filled else 0
	return grid.reshape(9, 9), probs.reshape(9, 9, 9), confidence


def read_puzzle(img, model):
	corners, method = find_grid(img)
	if corners is None:
		return None
	warped, H, H_inv = warp_grid(img, corners)
	binary = binarize_grid(warped)

	best = None
	for k, flip in ORIENTATIONS:
		grid, probs, confidence = read_cells(orient(binary, k, flip), model)
		if best is None or confidence > best["confidence"]:
			best = {"grid": grid, "probs": probs, "confidence": confidence, "orientation": (k, flip)}
	best.update(corners=corners, method=method, warped=warped, H_inv=H_inv)
	return best


def solve_with_retries(grid, probs, max_tries=10):
	if count_solutions(grid) == 1:
		return grid, solve(grid), None

	filled = np.argwhere(grid > 0)
	confidence = [probs[r, c].max() for r, c in filled]
	for idx in np.argsort(confidence)[:max_tries]:
		r, c = filled[idx]
		second = int(np.argsort(probs[r, c])[-2]) + 1
		changed = grid.copy()
		changed[r, c] = second
		if count_solutions(changed) == 1:
			return changed, solve(changed), (int(r), int(c), int(grid[r, c]), second)
	return grid, solve(grid), None


def draw_solution(img, result, givens, solved, size=450):
	k, flip = result["orientation"]
	givens_w = unorient(givens, k, flip)
	solved_w = unorient(solved, k, flip)
	step = size / 9

	out = img.copy()
	for r in range(9):
		for c in range(9):
			if givens_w[r, c] != 0:
				continue
			cell = np.array([[[c * step, r * step], [(c + 1) * step, (r + 1) * step],
							  [(c + 0.5) * step, (r + 0.5) * step]]], dtype=np.float32)
			tl, br, center = cv2.perspectiveTransform(cell, result["H_inv"])[0]
			cell_size = np.linalg.norm(br - tl) / np.sqrt(2)
			scale = cell_size / 45
			thickness = max(1, int(cell_size / 15))
			text = str(solved_w[r, c])
			(tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
			pos = (int(center[0] - tw / 2), int(center[1] + th / 2))
			cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 140, 0), thickness, cv2.LINE_AA)
	return out


def run(path, model):
	img = load_photo(path)
	result = read_puzzle(img, model)
	if result is None:
		return None, None
	givens, solved, changed = solve_with_retries(result["grid"], result["probs"])
	result.update(givens=givens, solved=solved, changed=changed)
	if solved is None:
		return result, None
	return result, draw_solution(img, result, givens, solved)


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("usage: python reader.py photo.jpeg")
		sys.exit(1)
	path = Path(sys.argv[1])
	result, drawn = run(path, load_model())
	if result is None:
		print("could not find the grid")
		sys.exit(1)

	print("orientation (rot90 k, flip):", result["orientation"], "| grid found with", result["method"])
	print("digits read:\n", result["grid"])
	if result["changed"]:
		r, c, old, new = result["changed"]
		print(f"grid was not solvable, changed cell ({r}, {c}) from {old} to {new}")
	if drawn is None:
		print("could not solve the puzzle")
		sys.exit(1)

	print("solution:\n", result["solved"])
	show_images([load_photo(path), drawn], ["input", "solved"], cols=2, save_as=f"solved_{path.stem}.png")
	print(f"saved results/solved_{path.stem}.png")
