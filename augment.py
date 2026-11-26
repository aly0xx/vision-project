# Makes more test photos from the original puzzle photos by rotating them and flipping them left to right.
# The hand-marked grid corners are moved with the same transform so grid detection can be checked.

import cv2
import numpy as np

from utils import DATA_DIR, PHOTOS_DIR, load_photo


ANGLES = [0, -30, -20, -10, 10, 20, 30, 90, 180]
CORNERS_DIR = DATA_DIR / "corners"


def load_corners(name):
	return np.loadtxt(CORNERS_DIR / f"{name}.txt", dtype=np.float32)


def save_corners(name, corners):
	np.savetxt(CORNERS_DIR / f"{name}.txt", corners, fmt="%.1f")


def flip_photo(img, corners):
	w = img.shape[1]
	flipped = cv2.flip(img, 1)
	new_corners = corners.copy()
	new_corners[:, 0] = w - 1 - corners[:, 0]
	return flipped, new_corners


def rotate_photo(img, corners, angle):
	h, w = img.shape[:2]
	M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
	cos, sin = abs(M[0, 0]), abs(M[0, 1])
	new_w = int(h * sin + w * cos)
	new_h = int(h * cos + w * sin)
	M[0, 2] += new_w / 2 - w / 2
	M[1, 2] += new_h / 2 - h / 2

	paper = [int(v) for v in np.median(img.reshape(-1, 3), axis=0)]
	rotated = cv2.warpAffine(img, M, (new_w, new_h), borderValue=paper)
	ones = np.ones((4, 1), np.float32)
	new_corners = np.hstack([corners, ones]) @ M.T
	return rotated, new_corners.astype(np.float32)


def make_variants(name):
	img = load_photo(PHOTOS_DIR / f"{name}.jpeg")
	corners = load_corners(name)
	made = []
	for flip in [False, True]:
		base, base_corners = flip_photo(img, corners) if flip else (img, corners)
		for angle in ANGLES:
			if angle == 0 and not flip:
				continue
			out, out_corners = rotate_photo(base, base_corners, angle)
			out_name = f"{name}_{'flip_' if flip else ''}rot{angle}"
			cv2.imwrite(str(PHOTOS_DIR / f"{out_name}.jpeg"), out)
			save_corners(out_name, out_corners)
			made.append(out_name)
	return made


if __name__ == "__main__":
	made = make_variants("puzzle01")
	print(f"made {len(made)} photos:", ", ".join(made))
