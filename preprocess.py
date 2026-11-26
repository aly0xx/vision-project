# Step 1: grayscale, Gaussian blur and adaptive threshold so grid lines and digits are white on black.
# Running it compares adaptive with global and Otsu thresholding.

import cv2

from utils import PHOTOS_DIR, load_photo, show_images


def to_gray(img):
	if img.ndim == 3:
		return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img


def threshold_photo(img, blur_size=5, block_size=15, c=8):
	gray = to_gray(img)
	blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
	binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
								   cv2.THRESH_BINARY_INV, block_size, c)
	return binary


def compare_thresholds(img):
	gray = cv2.GaussianBlur(to_gray(img), (5, 5), 0)
	_, global_th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
	_, otsu_th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	adaptive_th = threshold_photo(img)
	return global_th, otsu_th, adaptive_th


if __name__ == "__main__":
	img = load_photo(PHOTOS_DIR / "puzzle01.jpeg")
	global_th, otsu_th, adaptive_th = compare_thresholds(img)
	show_images([img, global_th, otsu_th, adaptive_th],
				["original", "global (T=127)", "Otsu", "adaptive Gaussian"],
				cols=2, save_as="step1_thresholds.png")
	print("saved results/step1_thresholds.png")
