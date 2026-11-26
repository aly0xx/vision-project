# Reproduces the failure cases we hit while building the pipeline (for the report), each with a figure:
# Hough corners on a curved page, a broken stroke, a digit cut by the cell margin, and a mirrored photo.

import cv2
import matplotlib.pyplot as plt
import numpy as np

from augment import load_corners
from cells import ORIENTATIONS, binarize_grid, clean_cell, find_digit, normalize_digit, orient, red_ink_mask
from grid import corners_from_contour, corners_from_hough, draw_corners, order_corners
from preprocess import threshold_photo
from reader import load_model, read_cells
from utils import PHOTOS_DIR, RESULTS_DIR, load_photo, show_images
from warp import split_cells, warp_grid


def predict(model, digit):
	probs = model.predict_proba(digit.reshape(1, -1) / 255.0)[0]
	return int(np.argmax(probs)) + 1, probs.max()


def hough_on_curved_page():
	name = "puzzle01_rot-20"
	img = load_photo(PHOTOS_DIR / f"{name}.jpeg")
	binary = threshold_photo(img)
	true = order_corners(load_corners(name))
	hough = order_corners(corners_from_hough(binary)[0])
	contour = order_corners(corners_from_contour(binary))
	out = draw_corners(draw_corners(draw_corners(img, true, (255, 0, 0)), hough, (0, 0, 255)), contour, (0, 200, 0))
	show_images([out, cv2.resize(out[:120, :180], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)],
				["true (blue), Hough (red), contour (green)", "zoom on the top-left corner"],
				cols=2, save_as="failure1_hough_curved_page.png")


def old_binary(img):
	corners = order_corners(corners_from_contour(threshold_photo(img)))
	warped, _, _ = warp_grid(img, corners)
	binary = threshold_photo(warped)
	binary[red_ink_mask(warped) > 0] = 0
	return warped, binary


def broken_stroke(model):
	img = load_photo(PHOTOS_DIR / "puzzle01.jpeg")
	_, binary = old_binary(img)
	cell = clean_cell(split_cells(binary, margin=0.1)[22])
	n, labels, stats, _ = cv2.connectedComponentsWithStats(cell)
	main = max(range(1, n), key=lambda i: stats[i][4])
	only_main = normalize_digit((labels == main).astype(np.uint8) * 255, stats[main])
	merged = normalize_digit(*find_digit(cell))
	d1, p1 = predict(model, only_main)
	d2, p2 = predict(model, merged)
	show_images([cell, only_main, merged],
				["cell (2, 4), 10% margin, lines kept", f"largest component only -> {d1} ({p1:.2f})",
				 f"broken pieces joined -> {d2} ({p2:.2f})"],
				cols=3, save_as="failure2_broken_stroke.png")


def digit_cut_by_margin(model):
	name = "puzzle01_rot-20"
	warped, old = old_binary(load_photo(PHOTOS_DIR / f"{name}.jpeg"))
	old_cell = split_cells(old, margin=0.1)[14]
	new_cell = split_cells(binarize_grid(warped))[14]

	images, titles = [], []
	for label, cell in [("10% margin, lines kept", old_cell), ("lines removed, 5% margin", new_cell)]:
		digit = normalize_digit(*find_digit(clean_cell(cell)))
		d, p = predict(model, digit)
		images += [cell, digit]
		titles += [label, f"read as {d} ({p:.2f}), true 9"]
	show_images(images, titles, cols=4, save_as="failure3_digit_cut_by_margin.png")


def mirrored_photo(model):
	name = "puzzle01_flip_rot0"
	img = load_photo(PHOTOS_DIR / f"{name}.jpeg")
	corners = order_corners(corners_from_contour(threshold_photo(img)))
	warped, _, _ = warp_grid(img, corners)
	binary = binarize_grid(warped)
	scores = [read_cells(orient(binary, k, flip), model)[2] for k, flip in ORIENTATIONS]

	fig, axes = plt.subplots(1, 2, figsize=(11, 4))
	axes[0].imshow(binary, cmap="gray")
	axes[0].set_title("mirrored photo after the warp")
	axes[0].axis("off")
	labels = [f"rot {90 * k}{' + flip' if flip else ''}" for k, flip in ORIENTATIONS]
	axes[1].bar(labels, scores, color=["#999999" if s < max(scores) else "#2a7" for s in scores])
	axes[1].set_ylabel("mean classifier confidence")
	axes[1].set_title("confidence for each of the 8 orientations")
	axes[1].tick_params(axis="x", rotation=45)
	fig.tight_layout()
	fig.savefig(RESULTS_DIR / "failure4_mirrored_photo.png", dpi=120)
	plt.close(fig)

	grid, _, _ = read_cells(binary, model)
	print("mirrored photo read without the orientation search:\n", grid)


if __name__ == "__main__":
	RESULTS_DIR.mkdir(exist_ok=True)
	model = load_model()
	hough_on_curved_page()
	broken_stroke(model)
	digit_cut_by_margin(model)
	mirrored_photo(model)
	print("saved results/failure1-4 figures")
