"""
This script implements an automated image separation system that categorizes images based on their filenames and moves
them into appropriate subfolders. The system processes images from a specified folder, where images follow a naming
convention of "ImageX" (where X is a number). Images are classified into two categories: Bright Field images (where X
is 0 or an even number) and Dark Field images (where X is an odd number). The script moves (rather than copies) the
images into two separate folders named "Bright Field" and "Dark Field", which are created at the same directory level
as the original folder. The implementation uses regular expressions for pattern matching and includes proper directory
creation and file movement operations. This utility is particularly useful for organizing microscopy or imaging data
where alternating images represent different illumination conditions.
"""

import re
import shutil
from pathlib import Path

RX = re.compile(r"^Image(\d+)\.[^.]+$", re.IGNORECASE)

def main():
    # Enter the path to your images folder here
    src_path = r"C:\Users\Andrea\Desktop\Data_First_Version\Camera"
    src = Path(src_path)
    if not src.is_dir():
        print(f"Error: {src} is not a valid folder.")
        return

    out_base = src.parent
    bright_dir = out_base / "Camera Bright Field"
    dark_dir   = out_base / "Camera Dark Field"
    bright_dir.mkdir(parents=True, exist_ok=True)
    dark_dir.mkdir(parents=True, exist_ok=True)

    moved_b = moved_d = 0
    for f in src.iterdir():
        if not f.is_file():
            continue
        m = RX.match(f.name)
        if not m:
            continue
        n = int(m.group(1))
        dst = (bright_dir if n % 2 == 0 else dark_dir) / f.name
        shutil.move(str(f), dst)
        if n % 2 == 0:
            moved_b += 1
        else:
            moved_d += 1

    print("Operation completed.")
    print(f"Bright (even): {moved_b}  -> {bright_dir}")
    print(f"Dark   (odd): {moved_d}  -> {dark_dir}")

if __name__ == "__main__":
    main()


