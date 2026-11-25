import cv2
import numpy as np

from preprocess import threshold_photo, to_gray
from utils import PHOTOS_DIR, load_photo, show_images


def find_lines(binary, vote_ratio=0.4):
	edges = cv2.Canny(binary, 100, 200)
	min_votes = int(vote_ratio * min(binary.shape))
	lines = cv2.HoughLines(edges, 1, np.pi / 180, min_votes)
	if lines is None:
		return edges, []
	return edges, [tuple(line[0]) for line in lines]


def split_lines(lines, angle_tol=20):
	horizontal = []
	vertical = []
	for rho, theta in lines:
		deg = np.degrees(theta)
		if abs(deg - 90) < angle_tol:
			horizontal.append((rho, theta))
		elif deg < angle_tol or deg > 180 - angle_tol:
			vertical.append((rho, theta))
	return horizontal, vertical


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
	horizontal, vertical = split_lines(lines)
	if len(horizontal) < 2 or len(vertical) < 2:
		return None, edges, lines

	h, w = binary.shape
	top = min(horizontal, key=lambda l: line_y_at(l, w / 2))
	bottom = max(horizontal, key=lambda l: line_y_at(l, w / 2))
	left = min(vertical, key=lambda l: line_x_at(l, h / 2))
	right = max(vertical, key=lambda l: line_x_at(l, h / 2))

	corners = [intersect(top, left), intersect(top, right),
			   intersect(bottom, right), intersect(bottom, left)]
	return np.array(corners, dtype=np.float32), edges, lines


def corners_from_contour(binary):
	contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	biggest = max(contours, key=cv2.contourArea)
	peri = cv2.arcLength(biggest, True)
	approx = cv2.approxPolyDP(biggest, 0.02 * peri, True)
	if len(approx) != 4:
		return None
	return approx.reshape(4, 2).astype(np.float32)


def order_corners(pts):
	pts = np.array(pts, dtype=np.float32)
	s = pts.sum(axis=1)
	d = np.diff(pts, axis=1).ravel()
	top_left = pts[np.argmin(s)]
	bottom_right = pts[np.argmax(s)]
	top_right = pts[np.argmin(d)]
	bottom_left = pts[np.argmax(d)]
	return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


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


def find_grid(img):
	binary = threshold_photo(img)
	corners, edges, lines = corners_from_hough(binary)
	method = "hough"
	if corners is None:
		corners = corners_from_contour(binary)
		method = "contour"
	if corners is None:
		return None, method
	return order_corners(corners), method


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
	_, edges, lines = corners_from_hough(binary)
	corners, method = find_grid(img)
	harris = harris_check(img, corners)

	print("method:", method)
	print("hough corners:\n", corners.round(1))
	print("harris corners:\n", harris.round(1))
	print("mean distance:", np.linalg.norm(corners - harris, axis=1).mean().round(2), "px")

	both = draw_corners(draw_corners(img, corners), harris, (255, 0, 0))
	show_images([edges, draw_lines(img, lines), both],
				["Canny edges", f"Hough lines ({len(lines)})", "corners: Hough (green), Harris (blue)"],
				cols=3, save_as="step2_grid.png")
	print("saved results/step2_grid.png")
