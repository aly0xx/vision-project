# Builds the digit datasets: synthetic digits for training and validation (split by font),
# and real digits cut from the photos (labelled with the ground truth) for testing only.

import re

import numpy as np

from cells import binarize_grid, process_cells, unorient
from grid import find_grid
from utils import DATA_DIR, PHOTOS_DIR, load_ground_truth, load_photo
from warp import split_cells, warp_grid


def known_orientation(name):
	flip = "_flip" in name
	match = re.search(r"rot(-?\d+)", name)
	angle = int(match.group(1)) if match else 0
	return round(angle / 90) % 4, flip


def upright_binary(img, name):
	corners, _ = find_grid(img)
	if corners is None:
		return None
	warped, _, _ = warp_grid(img, corners)
	k, flip = known_orientation(name)
	return unorient(binarize_grid(warped), k, flip)


def load_synthetic():
	data = np.load(DATA_DIR / "synthetic" / "digits.npz")
	X_train = data["X_train"].reshape(len(data["X_train"]), -1) / 255.0
	X_val = data["X_val"].reshape(len(data["X_val"]), -1) / 255.0
	return X_train, X_val, data["y_train"], data["y_val"]


def load_real(photos=None):
	if photos is None:
		photos = sorted(PHOTOS_DIR.glob("*.jpeg"))
	X, y, names = [], [], []
	for path in photos:
		binary = upright_binary(load_photo(path), path.stem)
		if binary is None:
			continue
		truth = [v for row in load_ground_truth(path.stem) for v in row]
		digits = process_cells(split_cells(binary))
		for digit, label in zip(digits, truth):
			if digit is not None and label != 0:
				X.append(digit.ravel() / 255.0)
				y.append(label)
				names.append(path.stem)
	return np.array(X), np.array(y), names


if __name__ == "__main__":
	X_train, X_val, y_train, y_val = load_synthetic()
	print("synthetic train:", X_train.shape, "val:", X_val.shape)
	X_test, y_test, names = load_real()
	print("real test digits:", X_test.shape, "from", len(set(names)), "photos")
	print("per digit:", np.bincount(y_test)[1:])
