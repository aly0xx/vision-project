# Photo Sudoku Reader

CSCE 4603 course project. Reads a printed Sudoku from a phone photo, solves it and draws the answer back on the photo.

## Setup

Uses the course venv (`../.venv`) with OpenCV, NumPy, Matplotlib and scikit-learn.

## Layout

- `data/photos/` puzzle photos (`puzzleNN.jpeg`)
- `data/ground_truth/` one text file per puzzle, 9 lines of 9 digits, `0` for empty (only printed givens, not pen marks)
- `results/` figures produced by the scripts (not committed)

## Running

Run from this folder with the course venv, e.g. `../.venv/bin/python run_checkpoint.py`.

- `preprocess.py` step 1, adaptive vs global vs Otsu threshold
- `grid.py` step 2, outer grid corners (Hough, Harris check, contour fallback)
- `warp.py` step 3, perspective warp and 81 cells
- `cells.py` step 4, cell cleaning, empty detection, 28x28 digit crops
- `synth_digits.py` generates `data/synthetic/digits.npz` (4500 digits 1-9)
- `run_checkpoint.py` runs steps 1-4 on every photo in `data/photos/`
