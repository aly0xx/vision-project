# Makes synthetic digits 1-9 from OpenCV Hershey fonts and free TrueType fonts, adds rotation, shear,
# blur and noise, then passes them through the same cell processing as the real photos.

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cells import clean_cell, find_digit, normalize_digit
from preprocess import threshold_photo
from utils import DATA_DIR, show_images


HERSHEY = {"hershey_simplex": cv2.FONT_HERSHEY_SIMPLEX, "hershey_duplex": cv2.FONT_HERSHEY_DUPLEX,
		   "hershey_complex": cv2.FONT_HERSHEY_COMPLEX, "hershey_triplex": cv2.FONT_HERSHEY_TRIPLEX,
		   "hershey_plain": cv2.FONT_HERSHEY_PLAIN}
FONT_FILES = {p.stem: p for p in sorted((DATA_DIR / "fonts").glob("*.[ot]tf"))}
VAL_FONTS = ["hershey_duplex", "NotoSans-Regular", "LiberationSerif-Regular", "URWGothic-Book"]
CELL_SIZE = 40
BIG = CELL_SIZE * 3


def draw_hershey(digit, font, height, paper, ink, rng):
	img = np.full((BIG, BIG), paper, np.uint8)
	thickness = int(rng.integers(1, 3))
	scale = cv2.getFontScaleFromHeight(font, height, thickness)
	(tw, th), _ = cv2.getTextSize(str(digit), font, scale, thickness)
	x = (BIG - tw) // 2 + int(rng.integers(-6, 7))
	y = (BIG + th) // 2 + int(rng.integers(-6, 7))
	cv2.putText(img, str(digit), (x, y), font, scale, ink, thickness, cv2.LINE_AA)
	return img


def draw_truetype(digit, path, height, paper, ink, rng):
	img = Image.new("L", (BIG, BIG), paper)
	draw = ImageDraw.Draw(img)
	font = ImageFont.truetype(str(path), 100)
	x0, y0, x1, y1 = draw.textbbox((0, 0), str(digit), font=font)
	font = ImageFont.truetype(str(path), int(100 * height / (y1 - y0)))
	x0, y0, x1, y1 = draw.textbbox((0, 0), str(digit), font=font)
	x = (BIG - (x1 - x0)) // 2 - x0 + int(rng.integers(-6, 7))
	y = (BIG - (y1 - y0)) // 2 - y0 + int(rng.integers(-6, 7))
	stroke = 1 if rng.random() < 0.2 else 0
	draw.text((x, y), str(digit), fill=ink, font=font, stroke_width=stroke, stroke_fill=ink)
	return np.array(img)


def render_digit(digit, font_name, rng):
	paper = int(rng.integers(170, 245))
	ink = int(rng.integers(0, 80))
	height = int(BIG * rng.uniform(0.6, 0.75))
	if font_name in HERSHEY:
		img = draw_hershey(digit, HERSHEY[font_name], height, paper, ink, rng)
	else:
		img = draw_truetype(digit, FONT_FILES[font_name], height, paper, ink, rng)

	angle = rng.uniform(-8, 8)
	shear = rng.uniform(-0.15, 0.15)
	M = cv2.getRotationMatrix2D((BIG / 2, BIG / 2), angle, 1.0)
	M[0, 0] *= rng.uniform(0.9, 1.2)
	M[0, 1] += shear
	img = cv2.warpAffine(img, M, (BIG, BIG), borderValue=paper)

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


def make_dataset(font_names, per_font_digit, rng):
	X, y = [], []
	for font_name in font_names:
		for digit in range(1, 10):
			count = 0
			while count < per_font_digit:
				out = preprocess_like_cells(render_digit(digit, font_name, rng))
				if out is not None:
					X.append(out)
					y.append(digit)
					count += 1
	return np.array(X), np.array(y)


if __name__ == "__main__":
	all_fonts = list(HERSHEY) + list(FONT_FILES)
	train_fonts = [f for f in all_fonts if f not in VAL_FONTS]
	rng = np.random.default_rng(0)
	X_train, y_train = make_dataset(train_fonts, 40, rng)
	X_val, y_val = make_dataset(VAL_FONTS, 40, rng)

	out_path = DATA_DIR / "synthetic" / "digits.npz"
	out_path.parent.mkdir(exist_ok=True)
	np.savez_compressed(out_path, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
	print(f"{len(train_fonts)} train fonts -> {len(X_train)} digits, {len(VAL_FONTS)} val fonts -> {len(X_val)} digits")

	rng = np.random.default_rng(1)
	fonts = ["hershey_simplex", "NimbusSans-Regular", "DejaVuSerif", "LiberationSans-Italic"]
	raw = [render_digit(d, f, rng) for f in fonts for d in range(1, 10)]
	show_images(raw, [f"{f[:12]} {d}" for f in fonts for d in range(1, 10)], cols=9, save_as="synthetic_raw.png")
	picks = [np.where(y_train == d)[0][i * 97] for d in range(1, 10) for i in range(3)]
	show_images([X_train[i] for i in picks], [str(y_train[i]) for i in picks], cols=9, save_as="synthetic_digits.png")
	print("saved results/synthetic_raw.png and results/synthetic_digits.png")
