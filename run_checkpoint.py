from cells import empty_accuracy, extract_digits
from grid import draw_corners, find_grid
from utils import DATA_DIR, PHOTOS_DIR, load_ground_truth, load_photo, show_images
from warp import draw_lattice, warp_grid


def run_photo(path):
	img = load_photo(path)
	corners, method = find_grid(img)
	if corners is None:
		print(f"{path.name}: grid not found")
		return False

	warped, _, _ = warp_grid(img, corners)
	digits, _, binary, _ = extract_digits(img)
	filled = sum(d is not None for d in digits)

	line = f"{path.name}: grid found ({method}), {filled} filled cells"
	if (DATA_DIR / "ground_truth" / f"{path.stem}.txt").exists():
		acc = empty_accuracy(digits, load_ground_truth(path.stem))
		line += f", empty/filled accuracy {acc:.3f}"
	print(line)

	show_images([draw_corners(img, corners), draw_lattice(warped), binary],
				["detected grid", "warped + lattice", "binary (red ink removed)"],
				cols=3, save_as=f"checkpoint_{path.stem}.png")
	return True


if __name__ == "__main__":
	photos = sorted(PHOTOS_DIR.glob("*.jp*g")) + sorted(PHOTOS_DIR.glob("*.png"))
	found = sum(run_photo(p) for p in photos)
	print(f"grid detection: {found}/{len(photos)} photos")
