# Dataset

## Layout

- `photos/` puzzle photos. `puzzle01.jpeg` is the original phone photo (343x295). The other 17 are made from it by `augment.py`: rotated by -30, -20, -10, 10, 20, 30, 90 and 180 degrees, and flipped left to right with the same rotations. The name says what was done, e.g. `puzzle01_flip_rot-20.jpeg`.
- `ground_truth/` one text file per puzzle (`puzzle01.txt`): 9 lines of 9 digits, `0` for an empty cell. Only the printed digits are givens; red pen marks are not. Copies of a puzzle use the same file, in the upright orientation.
- `corners/` the four outer grid corners for every photo (one `x y` pair per line). Marked by hand on `puzzle01.jpeg` and moved with the same rotation and flip for the copies.
- `fonts/` free fonts used to render the synthetic training digits (licences in `fonts/README.md`).
- `synthetic/digits.npz` made by `synth_digits.py` (not in the repo): `X_train`, `y_train`, `X_val`, `y_val`, 28x28 digit images after the same cell processing as the photos.

## Split

- Training: synthetic digits from 16 fonts (5760 digits).
- Validation: synthetic digits from 4 other fonts (1440 digits), used for all tuning.
- Test: the real printed digits cut from the photos (32 per photo, 576 in total). They are never used for training or tuning.

All 18 photos come from one printed puzzle, so the test set has only 32 different printed digits, each seen in 18 orientations.

## How the photo was taken

One phone photo of a printed puzzle, taken at a slight angle, so the grid is tilted and the page is a little curved at the top. Some answers were already written on it in red pen.

TODO (team): add the phone model, the lighting and where the puzzle came from.
