# Step 3: warp the grid to a square top-down view with a homography and cut it into 81 cells.

import cv2
import numpy as np

from grid import find_grid
from utils import PHOTOS_DIR, load_photo, show_images


def warp_grid(img, corners, size=450):
	dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
	H = cv2.getPerspectiveTransform(corners, dst)
	H_inv = np.linalg.inv(H)
	warped = cv2.warpPerspective(img, H, (size, size), flags=cv2.INTER_LINEAR)
	return warped, H, H_inv


def split_cells(warped, margin=0.05):
	size = warped.shape[0]
	step = size // 9
	m = int(step * margin)
	cells = []
	for row in range(9):
		for col in range(9):
			y0, x0 = row * step, col * step
			cell = warped[y0 + m:y0 + step - m, x0 + m:x0 + step - m]
			cells.append(cell)
	return cells


def draw_lattice(warped):
	out = warped.copy()
	size = out.shape[0]
	step = size // 9
	for i in range(10):
		cv2.line(out, (i * step, 0), (i * step, size), (0, 0, 255), 1)
		cv2.line(out, (0, i * step), (size, i * step), (0, 0, 255), 1)
	return out


def cells_figure(cells, save_as):
	show_images(cells, [str(i) for i in range(81)], cols=9, save_as=save_as)


if __name__ == "__main__":
	img = load_photo(PHOTOS_DIR / "puzzle01.jpeg")
	corners, method = find_grid(img)
	warped, H, H_inv = warp_grid(img, corners)
	cells = split_cells(warped)

	print("homography:\n", H.round(3))
	print("cells:", len(cells), "each", cells[0].shape)

	show_images([img, warped, draw_lattice(warped)],
				["original", "warped top-down", "9x9 lattice"],
				cols=3, save_as="step3_warp.png")
	cells_figure(cells, "step3_cells.png")
	print("saved results/step3_warp.png and results/step3_cells.png")
