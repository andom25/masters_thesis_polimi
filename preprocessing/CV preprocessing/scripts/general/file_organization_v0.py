"""
This script implements an automated file organization system that groups images in a specified folder into subfolders
based on filename prefixes. The prefix extraction algorithm identifies the part of the filename up to and including the
first numeric digit encountered. If no number is found in the filename, the entire filename (without extension) is used
as the prefix. The script creates subfolders for each unique prefix and moves the corresponding files into these
subfolders. This organization system is useful for grouping related images that share common naming patterns, such as
those generated from sequential processes or batch operations. The implementation uses regular expressions for pattern
matching and includes proper directory creation and file movement operations.
"""

import os
import shutil
import re

# Set your folder path here
input_folder = r"C:\Users\Andrea\Desktop\Final"

def extract_prefix(filename):
    # Find the first number in the filename and include that number in the prefix
    match = re.search(r'\d', filename)
    if match:
        index = match.start() + 1  # include the number itself
        return filename[:index]
    else:
        # If no number is found, take the whole name without the extension
        return os.path.splitext(filename)[0]

def split_images_by_prefix(input_folder):
    file_list = os.listdir(input_folder)
    
    # Keep only files (exclude folders)
    file_list = [f for f in file_list if os.path.isfile(os.path.join(input_folder, f))]

    prefixes = set()
    for file in file_list:
        prefix = extract_prefix(file)
        prefixes.add(prefix)

    for prefix in prefixes:
        subfolder = os.path.join(input_folder, prefix)
        os.makedirs(subfolder, exist_ok=True)

    for file in file_list:
        prefix = extract_prefix(file)
        src = os.path.join(input_folder, file)
        dst = os.path.join(input_folder, prefix)
        shutil.move(src, dst)

if __name__ == "__main__":
    
    if os.path.isdir(input_folder):
        split_images_by_prefix(input_folder)
        print(f"Images organized into subfolders in {input_folder}")
    else:
        print("Invalid folder path.")


