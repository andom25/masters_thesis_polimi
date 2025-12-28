"""
This script implements a batch image rotation utility for processing images contained in a specified folder.
The script supports three rotation modes: 90 degrees clockwise, 90 degrees counterclockwise, and 180 degrees.
The rotation is applied directly to the original images, modifying them in place. The implementation uses OpenCV's
rotation functions and supports multiple image formats (JPG, JPEG, PNG, TIF, TIFF, BMP, WEBP). The script includes
error handling for unreadable images and provides progress feedback during batch processing.
"""

from pathlib import Path
import cv2
from tqdm import tqdm

# Set folder path and rotation mode
FOLDER_PATH = Path(r"C:\Users\Andrea\Desktop\Data\Stitched Images")
ROTATION_MODE = 3  # 1 = 90° clockwise, 2 = 90° counterclockwise, 3 = 180°

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

ROTATION_MAP = {
    1: cv2.ROTATE_90_CLOCKWISE,
    2: cv2.ROTATE_90_COUNTERCLOCKWISE,
    3: cv2.ROTATE_180
}

def rotate_folder(folder: Path, rotate_flag: int):
    imgs = [p for p in folder.iterdir() if p.suffix.lower() in EXTS]
    rotated = 0
    for p in tqdm(imgs, desc="Rotating", unit="img"):
        try:
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"\nCould not read: {p.name}")
                continue
            img = cv2.rotate(img, rotate_flag)
            cv2.imwrite(str(p), img)
            rotated += 1
        except Exception as e:
            print(f"\nError with {p.name}: {e}")
    return rotated, len(imgs)

if __name__ == "__main__":
    folder = FOLDER_PATH
    if not folder.is_dir():
        print(f"❌ Invalid folder path: {folder}")
    elif ROTATION_MODE not in ROTATION_MAP:
        print(f"❌ Invalid rotation mode: {ROTATION_MODE}. Must be 1, 2, or 3.")
    else:
        rotate_flag = ROTATION_MAP[ROTATION_MODE]
        done, total = rotate_folder(folder, rotate_flag)
        print(f"\nImages found: {total}, rotated: {done}")
        print("Rotation executed successfully.")



