# Packs the photos, ground truth, corners and dataset README into sudoku_dataset.zip for the submission.

import zipfile

from utils import BASE_DIR, DATA_DIR


if __name__ == "__main__":
	out = BASE_DIR / "sudoku_dataset.zip"
	with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
		z.write(DATA_DIR / "README.md", "README.md")
		for folder in ["photos", "ground_truth", "corners"]:
			for path in sorted((DATA_DIR / folder).iterdir()):
				z.write(path, f"{folder}/{path.name}")
	print(f"saved {out.name}")
