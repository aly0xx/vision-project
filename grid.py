# Step 2: find the four outer corners of the grid from the largest contour, with Canny + Hough lines
# as a fallback. Each result is checked by warping it and testing that grid lines are where we expect.

import cv2
import numpy as np

from preprocess import threshold_photo, to_gray
from utils import PHOTOS_DIR, load_photo, show_images


def find_lines(binary, vote_ratio=0.3):
	edges = cv2.Canny(binary, 100, 200)
	min_votes = int(vote_ratio * min(binary.shape))
	lines = cv2.HoughLines(edges, 1, np.pi / 180, min_votes)
	if lines is None:
		return edges, []
	return edges, [tuple(line[0]) for line in lines]


def split_lines(lines, angle_tol=15):
	degs = [np.degrees(theta) % 90 for _, theta in lines]
	hist, _ = np.histogram(degs, bins=45, range=(0, 90))
	dominant = np.argmax(hist) * 2 + 1

	families = [[], []]
	for rho, theta in lines:
		deg = np.degrees(theta)
		for k, ref in enumerate([dominant, dominant + 90]):
			diff = (deg - ref + 90) % 180 - 90
			if abs(diff) < angle_tol:
				if abs(deg - ref) > 90:
					families[k].append((-rho, theta - np.pi))
				else:
					families[k].append((rho, theta))
	return families


def line_x_at(line, y):
	rho, theta = line
	return (rho - y * np.sin(theta)) / np.cos(theta)


def line_y_at(line, x):
	rho, theta = line
	return (rho - x * np.cos(theta)) / np.sin(theta)


def intersect(line1, line2):
	rho1, theta1 = line1
	rho2, theta2 = line2
	a = np.array([[np.cos(theta1), np.sin(theta1)],
				  [np.cos(theta2), np.sin(theta2)]])
	b = np.array([rho1, rho2])
	x, y = np.linalg.solve(a, b)
	return [x, y]


def corners_from_hough(binary):
	edges, lines = find_lines(binary)
	if len(lines) < 4:
		return None, edges, lines
	family_a, family_b = split_lines(lines)
	if len(family_a) < 2 or len(family_b) < 2:
		return None, edges, lines

	first_a = min(family_a, key=lambda l: l[0])
	last_a = max(family_a, key=lambda l: l[0])
	first_b = min(family_b, key=lambda l: l[0])
	last_b = max(family_b, key=lambda l: l[0])

	corners = [intersect(first_a, first_b), intersect(first_a, last_b),
			   intersect(last_a, last_b), intersect(last_a, first_b)]
	return np.array(corners, dtype=np.float32), edges, lines


def corners_from_contour(binary):
	contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	biggest = max(contours, key=cv2.contourArea)
	hull = cv2.convexHull(biggest)
	peri = cv2.arcLength(hull, True)
	for eps in np.arange(0.01, 0.1, 0.005):
		approx = cv2.approxPolyDP(hull, eps * peri, True)
		if len(approx) == 4:
			return approx.reshape(4, 2).astype(np.float32)
	return None


def order_corners(pts):
	pts = np.array(pts, dtype=np.float32)
	center = pts.mean(axis=0)
	angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
	pts = pts[np.argsort(angles)]
	start = np.argmin(pts.sum(axis=1))
	return np.roll(pts, -start, axis=0)


def grid_score(binary, corners, size=450, band=3):
	dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
	H = cv2.getPerspectiveTransform(corners, dst)
	warped = cv2.warpPerspective(binary, H, (size, size))
	step = size / 9
	on_lines = []
	off_lines = []
	for i in range(10):
		pos = int(min(i * step, size - band))
		lo = max(pos - band, 0)
		on_lines.append(warped[lo:pos + band, :].max(axis=0).mean())
		on_lines.append(warped[:, lo:pos + band].max(axis=1).mean())
	for i in range(9):
		pos = int((i + 0.5) * step)
		off_lines.append(warped[pos - band:pos + band, :].max(axis=0).mean())
		off_lines.append(warped[:, pos - band:pos + band].max(axis=1).mean())
	return (np.mean(on_lines) - np.mean(off_lines)) / 255


def harris_check(img, corners, window=10):
	gray = np.float32(to_gray(img))
	response = cv2.cornerHarris(gray, 3, 3, 0.04)
	h, w = gray.shape

	refined = []
	for x, y in corners:
		x0, x1 = max(int(x) - window, 0), min(int(x) + window, w)
		y0, y1 = max(int(y) - window, 0), min(int(y) + window, h)
		patch = response[y0:y1, x0:x1]
		if patch.size == 0:
			refined.append([x, y])
			continue
		py, px = np.unravel_index(np.argmax(patch), patch.shape)
		refined.append([x0 + px, y0 + py])
	return np.array(refined, dtype=np.float32)


def find_grid(img, min_score=0.3, binary=None):
	if binary is None:
		binary = threshold_photo(img)
	contour = corners_from_contour(binary)
	if contour is not None and grid_score(binary, order_corners(contour)) >= min_score:
		return order_corners(contour), "contour"

	hough, _, _ = corners_from_hough(binary)
	if hough is not None and grid_score(binary, order_corners(hough)) >= min_score:
		return order_corners(hough), "hough"
	return None, "none"


def draw_lines(img, lines):
	out = img.copy()
	for rho, theta in lines:
		a, b = np.cos(theta), np.sin(theta)
		x0, y0 = a * rho, b * rho
		p1 = (int(x0 - 2000 * b), int(y0 + 2000 * a))
		p2 = (int(x0 + 2000 * b), int(y0 - 2000 * a))
		cv2.line(out, p1, p2, (0, 0, 255), 1)
	return out


def draw_corners(img, corners, color=(0, 255, 0)):
	out = img.copy()
	pts = corners.astype(np.int32)
	cv2.polylines(out, [pts], True, color, 2)
	for x, y in pts:
		cv2.circle(out, (int(x), int(y)), 5, color, -1)
	return out


if __name__ == "__main__":
	img = load_photo(PHOTOS_DIR / "puzzle01.jpeg")
	binary = threshold_photo(img)
	hough, edges, lines = corners_from_hough(binary)
	corners = order_corners(hough)
	harris = harris_check(img, corners)

	print("hough corners:\n", corners.round(1))
	print("harris corners:\n", harris.round(1))
	print("mean distance:", np.linalg.norm(corners - harris, axis=1).mean().round(2), "px")

	both = draw_corners(draw_corners(img, corners), harris, (255, 0, 0))
	show_images([edges, draw_lines(img, lines), both],
				["Canny edges", f"Hough lines ({len(lines)})", "corners: Hough (green), Harris (blue)"],
				cols=3, save_as="step2_grid.png")
	print("saved results/step2_grid.png")
