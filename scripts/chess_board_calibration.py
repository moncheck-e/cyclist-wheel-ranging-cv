import cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

def load_images(image_dir: Path) -> tuple[list, list]:
    images, names = [], []
    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  Error: Could not load {img_path.name}")
            else:
                print(f"  Loaded {img_path.name}")
                images.append(img)
                names.append(img_path.name)
    print(f"  Successfully loaded {len(images)} images\n")
    return images, names

def find_chessboard_points(images: list, names: list, chessboard_size: tuple, square_size: float):
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints, imgpoints = [], []
    for name, img in zip(names, images):
        ret, corners = cv2.findChessboardCorners(img, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            print(f"  {name}: found")
        else:
            print(f"  {name}: NOT found")

    print(f"  Found chessboard in {len(objpoints)}/{len(images)} images\n")
    return objpoints, imgpoints

def calibrate(image_dir: Path, output_dir: Path, chessboard_size: tuple, square_size: float, label: str):
    print(f"Calibrating: {label}")

    images, names = load_images(image_dir)
    if not images:
        print("  No images loaded, skipping.\n")
        return

    objpoints, imgpoints = find_chessboard_points(images, names, chessboard_size, square_size)
    if not objpoints:
        print("  No chessboard detections, skipping calibration.\n")
        return

    h, w = images[0].shape[:2]
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None) # type: ignore

    print(f"  Camera Matrix:\n{camera_matrix}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f'camera_matrix_{label}.npy', camera_matrix)
    np.save(output_dir / f'dist_coeffs_{label}.npy', dist_coeffs)
    print(f"  Saved camera_matrix_{label}.npy and dist_coeffs_{label}.npy\n")

# Run calibrations
CHESSBOARD_SIZE = (5, 3)        # (columns, rows) internal corners
SQUARE_SIZE     = 0.036    # size of chessboard square in meters
OUTPUT_DIR      = PROJECT_ROOT / "data"

calibrate(
    image_dir      = PROJECT_ROOT / "calibration" / "chessboard_1x_zoom",
    output_dir     = OUTPUT_DIR,
    chessboard_size = CHESSBOARD_SIZE,
    square_size    = SQUARE_SIZE,
    label          = '1x'
)

calibrate(
    image_dir      = PROJECT_ROOT / "calibration" / "chessboard_0.5x_zoom",
    output_dir     = OUTPUT_DIR,
    chessboard_size = CHESSBOARD_SIZE,
    square_size    = SQUARE_SIZE,
    label          = '0x5'
)