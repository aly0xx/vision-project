# Step 4: clean each cell with morphology and use connected components to decide empty or digit.
# Digits are cropped, padded and resized to 28x28. Red pen marks are ignored.

import cv2
import numpy as np

from grid import find_grid
from preprocess import threshold_photo
from utils import PHOTOS_DIR, load_ground_truth, load_photo, show_images
from warp import split_cells, warp_grid


DIGIT_SIZE = 28


def red_ink_mask(img, min_diff=15):
	b, g, r = cv2.split(img.astype(np.int16))
	mask = ((r - g) > min_diff).astype(np.uint8) * 255
	return cv2.dilate(mask, np.ones((5, 5), np.uint8))


def clean_cell(cell_binary, kernel_size=2):
	kernel = np.ones((kernel_size, kernel_size), np.uint8)
	return cv2.morphologyEx(cell_binary, cv2.MORPH_OPEN, kernel)


def find_digit(cell_binary, min_area=0.02, max_area=0.6):
	h, w = cell_binary.shape
	n, labels, stats, centroids = cv2.connectedComponentsWithStats(cell_binary, connectivity=8)

	best = None
	best_dist = None
	for i in range(1, n):
		x, y, bw, bh, area = stats[i]
		if area < min_area * h * w or area > max_area * h * w:
			continue
		if bw > 0.95 * w or bh > 0.95 * h:
			continue
		if max(bw, bh) < 0.3 * h:
			continue
		cx, cy = centroids[i]
		if abs(cx - w / 2) > 0.3 * w or abs(cy - h / 2) > 0.3 * h:
			continue
		dist = (cx - w / 2) ** 2 + (cy - h / 2) ** 2
		if best is None or dist < best_dist:
			best = i
			best_dist = dist

	if best is None:
		return None
	return (labels == best).astype(np.uint8) * 255, stats[best]


def normalize_digit(digit_mask, stat, size=DIGIT_SIZE, pad=4):
	x, y, w, h, _ = stat
	crop = digit_mask[y:y + h, x:x + w]
	side = max(w, h)
	square = np.zeros((side, side), np.uint8)
	square[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop

	inner = size - 2 * pad
	small = cv2.resize(square, (inner, inner), interpolation=cv2.INTER_AREA)
	out = np.zeros((size, size), np.uint8)
	out[pad:pad + inner, pad:pad + inner] = small
	return out


def process_cells(binary_cells):
	digits = []
	for cell in binary_cells:
		cleaned = clean_cell(cell)
		found = find_digit(cleaned)
		if found is None:
			digits.append(None)
		else:
			digits.append(normalize_digit(*found))
	return digits


def extract_digits(img):
	corners, _ = find_grid(img)
	warped, H, H_inv = warp_grid(img, corners)
	binary = threshold_photo(warped)
	binary[red_ink_mask(warped) > 0] = 0
	binary_cells = split_cells(binary)
	return process_cells(binary_cells), warped, binary, H_inv


def empty_accuracy(digits, truth):
	flat = [v for row in truth for v in row]
	correct = sum((d is None) == (t == 0) for d, t in zip(digits, flat))
	return correct / 81


if __name__ == "__main__":
	img = load_photo(PHOTOS_DIR / "puzzle01.jpeg")
	digits, warped, binary, _ = extract_digits(img)
	truth = load_ground_truth("puzzle01")

	print("filled cells found:", sum(d is not None for d in digits))
	print("filled cells in ground truth:", sum(v != 0 for row in truth for v in row))
	print(f"empty/filled accuracy: {empty_accuracy(digits, truth):.3f}")

	example = 12
	raw_cell = split_cells(binary)[example]
	cleaned = clean_cell(raw_cell)
	n, labels = cv2.connectedComponents(cleaned)
	label_view = np.uint8(labels * (255 / max(n - 1, 1)))
	show_images([raw_cell, cleaned, label_view, digits[example]],
				["cell threshold", "after opening", f"components ({n - 1})", "28x28 digit"],
				cols=4, save_as="step4_cell_example.png")

	blank = np.zeros((DIGIT_SIZE, DIGIT_SIZE), np.uint8)
	show_images([d if d is not None else blank for d in digits], [str(i) for i in range(81)],
				cols=9, save_as="step4_digits.png")
	print("saved results/step4_cell_example.png and results/step4_digits.png")
