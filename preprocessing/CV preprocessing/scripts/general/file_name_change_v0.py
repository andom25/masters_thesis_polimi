"""
This script implements a batch file renaming utility for PNG files in a specified folder. The script processes all PNG
files in the target directory and renames them by extracting only the last 3 characters of their original filename
(before the file extension). This operation is performed in-place, modifying the original filenames. The implementation
includes proper file extension handling and provides feedback for each renamed file. This utility is particularly useful
for standardizing filenames or extracting specific identifier patterns from existing file naming conventions.
"""

import os

# Path
input_folder = r"C:\Users\Andrea\Desktop\Data\Netfabb Slices"

def rename_files_in_folder(folder):
    # Iterate through all files in the folder
    for filename in os.listdir(folder):
        full_path = os.path.join(folder, filename)
        
        # Check if it's a PNG file
        if os.path.isfile(full_path) and filename.lower().endswith(".png"):
            name_without_extension = os.path.splitext(filename)[0]
            extension = os.path.splitext(filename)[1]

            # Take the last 3 characters of the name (before the extension)
            new_name = name_without_extension[-3:] + extension
            new_path = os.path.join(folder, new_name)

            # Rename the file
            os.rename(full_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")

# Example usage
rename_files_in_folder(input_folder)

