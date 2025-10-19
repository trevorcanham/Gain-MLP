import cv2
import os
import numpy as np
import pdb

def compare_image_directories(dir1_path, dir2_path):
    """
    Loads and compares images from two specified directories.
    Prints whether corresponding images are identical or different,
    and saves difference images if they are different.
    """
    files1 = sorted([f for f in os.listdir(dir1_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))])
    files2 = sorted([f for f in os.listdir(dir2_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))])

    # Iterate through files in both directories, assuming a one-to-one correspondence
    # You might need more complex logic if file names don't directly match or if
    # you want to compare all pairs.
    for filename1, filename2 in zip(files1, files2):
        img1_path = os.path.join(dir1_path, filename1)
        img2_path = os.path.join(dir2_path, filename2)

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None:
            print(f"Error: Could not load image from {img1_path}")
            continue
        if img2 is None:
            print(f"Error: Could not load image from {img2_path}")
            continue

        # Check if images have the same dimensions and channels
        if img1.shape == img2.shape:
            difference = cv2.subtract(img1, img2)
            b, g, r = cv2.split(difference)

            # Check if all channels are completely black (no difference)
            if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
                print(f"Images {filename1} and {filename2} are identical.")
            else:
                print(f"Images {filename1} and {filename2} are different.")
                # Optionally save the difference image
                diff_filename = f"diff_{filename1}"
                cv2.imwrite(os.path.join("differences", diff_filename), difference)
                print(f"Difference saved as {os.path.join('differences', diff_filename)}")
        else:
            print(f"Images {filename1} and {filename2} have different shapes.")

if __name__ == "__main__":
    # Create a directory for saving difference images if it doesn't exist
    if not os.path.exists("differences"):
        os.makedirs("differences")

    # Example usage: Replace with your actual directory paths
    directory1 = '/home/tcanham/Gain-MLP/models/20251018_072834/recon/'
    directory2 = 'output/'

    compare_image_directories(directory1, directory2)