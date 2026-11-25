# Photo Sudoku Reader

CSCE 4603 course project. Reads a printed Sudoku from a phone photo, solves it and draws the answer back on the photo.

## Setup

Uses the course venv (`../.venv`) with OpenCV, NumPy, Matplotlib and scikit-learn.

## Layout

- `data/photos/` puzzle photos (`puzzleNN.jpeg`)
- `data/ground_truth/` one text file per puzzle, 81 characters, `0` for empty
- `results/` figures produced by the scripts (not committed)
