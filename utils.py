# Shared helpers: project paths, loading photos and ground truth, and saving labelled figures.

from pathlib import Path

import cv2
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PHOTOS_DIR = DATA_DIR / "photos"
RESULTS_DIR = BASE_DIR / "results"


def load_photo(path):
	img = cv2.imread(str(path))
	if img is None:
		raise FileNotFoundError(f"could not read {path}")
	return img


def to_rgb(img):
	if img.ndim == 2:
		return img
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_images(images, titles, cols=3, save_as=None):
	rows = (len(images) + cols - 1) // cols
	fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
	axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

	for ax in axes:
		ax.axis("off")
	for ax, img, title in zip(axes, images, titles):
		ax.imshow(to_rgb(img), cmap="gray")
		ax.set_title(title)

	fig.tight_layout()
	if save_as is not None:
		RESULTS_DIR.mkdir(exist_ok=True)
		fig.savefig(RESULTS_DIR / save_as, dpi=120, bbox_inches="tight")
	plt.close(fig)


def load_ground_truth(name):
	puzzle = name.split("_")[0]
	path = DATA_DIR / "ground_truth" / f"{puzzle}.txt"
	digits = [int(ch) for ch in path.read_text() if ch.isdigit()]
	return [digits[i * 9:(i + 1) * 9] for i in range(9)]
