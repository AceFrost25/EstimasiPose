import os
import sys

# Define the desired working directory (replace with your actual path)
target_directory = "/home/zephyr/vision/DenseFusion"

# Change the current working directory
os.chdir(target_directory)

# Insert the new working directory at the beginning of sys.path
sys.path.insert(0, os.getcwd())
