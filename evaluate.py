# Computes all the numbers for the report: grid detection rate (and method comparisons), empty-cell accuracy,
# classifier test accuracy, puzzles solved end to end, and a log of everything that went wrong.

import pickle

import cv2
import numpy as np
from sklearn.metrics import accuracy_score

from augment import load_corners
from cells import orient, process_cells, unorient
from classify import MODELS_PATH
from dataset import known_orientation, load_real, upright_binary
from grid import corners_from_contour, corners_from_hough, find_grid, harris_check, order_corners
from preprocess import threshold_photo, to_gray
from reader import load_model, run
from solver import solve
from utils import PHOTOS_DIR, RESULTS_DIR, load_ground_truth, load_photo, show_images
from warp import split_cells


TOLERANCE = 0.25


def corner_error(found, true):
	if found is None:
		return np.inf
	found = order_corners(found)
	cell = np.linalg.norm(true[1] - true[0]) / 9
	return min(np.linalg.norm(np.roll(found, k, axis=0) - true, axis=1).max() for k in range(4)) / cell


def same_orientation(read, known):
	test = np.arange(81).reshape(9, 9)
	return np.array_equal(orient(test, *read), unorient(test, *known))


def otsu(img):
	gray = cv2.GaussianBlur(to_gray(img), (5, 5), 0)
	return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def grid_detection(photos):
	errors = {"final (contour, Hough fallback)": [], "contour only": [], "Hough only": [],
			  "Hough + Harris": [], "final with Otsu threshold": []}
	for path in photos:
		img = load_photo(path)
		true = order_corners(load_corners(path.stem))
		binary = threshold_photo(img)
		hough, _, _ = corners_from_hough(binary)
		errors["final (contour, Hough fallback)"].append(corner_error(find_grid(img)[0], true))
		errors["contour only"].append(corner_error(corners_from_contour(binary), true))
		errors["Hough only"].append(corner_error(hough, true))
		harris = harris_check(img, order_corners(hough)) if hough is not None else None
		errors["Hough + Harris"].append(corner_error(harris, true))
		errors["final with Otsu threshold"].append(corner_error(find_grid(img, binary=otsu(img))[0], true))

	lines = [f"Grid detection ({len(photos)} photos, detected = every corner within {TOLERANCE} cell of the hand-marked corner)"]
	for name, errs in errors.items():
		errs = np.array(errs)
		found = errs[np.isfinite(errs)]
		mean = f"{found.mean():.2f}" if len(found) else "-"
		lines.append(f"  {name:34s} {np.sum(errs < TOLERANCE):2d}/{len(errs)}  mean error {mean} cells")
	return lines


def empty_cells(photos):
	results = {"with opening": [], "without opening": []}
	for path in photos:
		binary = upright_binary(load_photo(path), path.stem)
		truth = np.array(load_ground_truth(path.stem)).ravel()
		for name, morph in [("with opening", True), ("without opening", False)]:
			digits = process_cells(split_cells(binary), use_morphology=morph)
			results[name].append(np.mean([(d is None) == (t == 0) for d, t in zip(digits, truth)]))
	lines = ["Empty / filled cell accuracy (all 81 cells of every photo)"]
	for name, accs in results.items():
		lines.append(f"  {name:16s} {np.mean(accs):.4f}  (worst photo {np.min(accs):.4f})")
	return lines


def classifiers():
	with open(MODELS_PATH, "rb") as f:
		models = pickle.load(f)["models"]
	X_test, y_test, _ = load_real()
	lines = [f"Digit accuracy on {len(y_test)} real test digits (trained on synthetic only)"]
	for name, model in models.items():
		lines.append(f"  {name:10s} {accuracy_score(y_test, model.predict(X_test)):.3f}")
	return lines


def end_to_end(photos):
	model = load_model()
	truth_solution = {}
	lines = ["End to end"]
	failures = []
	solved_ok = 0
	retried = 0
	for path in photos:
		truth = np.array(load_ground_truth(path.stem))
		puzzle = path.stem.split("_")[0]
		if puzzle not in truth_solution:
			truth_solution[puzzle] = solve(truth)

		result, drawn = run(path, model)
		if result is None:
			failures.append(f"{path.stem}: grid not found")
			continue
		if not same_orientation(result["orientation"], known_orientation(path.stem)):
			failures.append(f"{path.stem}: picked orientation {result['orientation']}, "
							f"true {known_orientation(path.stem)}")
		wrong = np.argwhere(result["grid"] != truth)
		for r, c in wrong:
			failures.append(f"{path.stem}: cell ({r}, {c}) read as {result['grid'][r, c]}, true {truth[r, c]}")
		if result["changed"]:
			retried += 1
		if result["solved"] is not None and np.array_equal(result["solved"], truth_solution[puzzle]):
			solved_ok += 1
		else:
			failures.append(f"{path.stem}: not solved correctly")

	lines.append(f"  puzzles solved correctly: {solved_ok}/{len(photos)}")
	lines.append(f"  solved only after changing the least sure digit: {retried}")
	return lines, failures


if __name__ == "__main__":
	photos = sorted(PHOTOS_DIR.glob("*.jpeg"))
	lines = grid_detection(photos) + [""] + empty_cells(photos) + [""] + classifiers() + [""]
	e2e, failures = end_to_end(photos)
	lines += e2e + ["", f"Problems found ({len(failures)}):"] + ["  " + f for f in failures]

	text = "\n".join(lines)
	print(text)
	RESULTS_DIR.mkdir(exist_ok=True)
	(RESULTS_DIR / "evaluation.txt").write_text(text + "\n")
	print("\nsaved results/evaluation.txt")
