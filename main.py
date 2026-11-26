# Runs the whole project from start to end: figures for steps 1-4, synthetic data, training,
# evaluation, failure cases, and the solved answer drawn on every photo in data/photos.

import runpy

from reader import load_model, run
from utils import BASE_DIR, PHOTOS_DIR, load_photo, show_images


SCRIPTS = ["preprocess.py", "grid.py", "warp.py", "cells.py", "synth_digits.py",
		   "classify.py", "evaluate.py", "failures.py"]


def solve_all():
	model = load_model()
	solved = 0
	photos = sorted(PHOTOS_DIR.glob("*.jpeg"))
	for path in photos:
		result, drawn = run(path, model)
		if drawn is None:
			print(f"{path.name}: not solved")
			continue
		solved += 1
		show_images([load_photo(path), drawn], ["input", "solved"], cols=2, save_as=f"solved_{path.stem}.png")
	print(f"solved {solved}/{len(photos)} photos, saved results/solved_*.png")


if __name__ == "__main__":
	for script in SCRIPTS:
		print("=" * 70)
		print(f"Running {script}")
		print("=" * 70)
		runpy.run_path(str(BASE_DIR / script), run_name="__main__")

	print("=" * 70)
	print("Solving every photo")
	print("=" * 70)
	solve_all()
