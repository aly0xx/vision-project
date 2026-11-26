# Makes synthetic training digits 1-9 with cv2.putText, adds rotation, shear, blur and noise,
# then passes them through the same cell processing as the real photos.

import cv2
import numpy as np

from cells import clean_cell, find_digit, normalize_digit
from preprocess import threshold_photo
from utils import DATA_DIR, show_images


FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
		 cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_PLAIN]
CELL_SIZE = 40


def render_digit(digit, rng):
	big = CELL_SIZE * 3
	paper = int(rng.integers(170, 245))
	ink = int(rng.integers(0, 80))
	img = np.full((big, big), paper, np.uint8)

	font = FONTS[rng.integers(len(FONTS))]
	if rng.random() < 0.2:
		font |= cv2.FONT_ITALIC
	thickness = int(rng.integers(1, 3))
	scale = cv2.getFontScaleFromHeight(font, int(big * rng.uniform(0.6, 0.75)), thickness)

	(tw, th), _ = cv2.getTextSize(str(digit), font, scale, thickness)
	x = (big - tw) // 2 + int(rng.integers(-6, 7))
	y = (big + th) // 2 + int(rng.integers(-6, 7))
	cv2.putText(img, str(digit), (x, y), font, scale, ink, thickness, cv2.LINE_AA)

	angle = rng.uniform(-8, 8)
	shear = rng.uniform(-0.15, 0.15)
	M = cv2.getRotationMatrix2D((big / 2, big / 2), angle, 1.0)
	M[0, 0] *= rng.uniform(1.0, 1.3)
	M[0, 1] += shear
	img = cv2.warpAffine(img, M, (big, big), borderValue=paper)

	img = cv2.resize(img, (CELL_SIZE, CELL_SIZE), interpolation=cv2.INTER_AREA)
	return add_noise(img, rng)


def add_noise(img, rng):
	img = img.astype(np.float32)
	if rng.random() < 0.5:
		img = cv2.GaussianBlur(img, (3, 3), 0)
	img += rng.normal(0, rng.uniform(2, 12), img.shape)
	if rng.random() < 0.3:
		spots = rng.random(img.shape)
		img[spots < 0.01] = 0
		img[spots > 0.99] = 255
	img = img * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)
	return np.clip(img, 0, 255).astype(np.uint8)


def preprocess_like_cells(cell):
	binary = threshold_photo(cell)
	found = find_digit(clean_cell(binary))
	if found is None:
		return None
	return normalize_digit(*found)


def make_dataset(per_digit=500, seed=0):
	rng = np.random.default_rng(seed)
	X, y = [], []
	skipped = 0
	for digit in range(1, 10):
		count = 0
		while count < per_digit:
			out = preprocess_like_cells(render_digit(digit, rng))
			if out is None:
				skipped += 1
				continue
			X.append(out)
			y.append(digit)
			count += 1
	return np.array(X), np.array(y), skipped


if __name__ == "__main__":
	X, y, skipped = make_dataset()
	out_path = DATA_DIR / "synthetic" / "digits.npz"
	out_path.parent.mkdir(exist_ok=True)
	np.savez_compressed(out_path, X=X, y=y)
	print(f"saved {len(X)} digits to {out_path.relative_to(DATA_DIR.parent)} (skipped {skipped} bad renders)")

	rng = np.random.default_rng(1)
	raw = [render_digit(d, rng) for d in range(1, 10)]
	picks = [np.where(y == d)[0][i] for d in range(1, 10) for i in range(3)]
	show_images(raw, [f"raw {d}" for d in range(1, 10)], cols=9, save_as="synthetic_raw.png")
	show_images([X[i] for i in picks], [str(y[i]) for i in picks], cols=9, save_as="synthetic_digits.png")
	print("saved results/synthetic_raw.png and results/synthetic_digits.png")
