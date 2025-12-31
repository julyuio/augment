import os
import cv2
import numpy as np

from .core import process_dataset, copy_boxes

# ---------------------------------------------------------
# CONVERT IMAGE TO GRAYSCALE (3-channel output)
# ---------------------------------------------------------
def convertGrayscale(img,factor=0):
    """
    Converts image to grayscale but returns 3-channel output
    so YOLO models can still train on it.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray_3ch



# ---------------------------------------------------------
# Main entry from __init__ 
# ---------------------------------------------------------
def convertGrayscale_main (root_dir, output_dir, debug=False, verbose=True):
    if verbose: 
        print(f'>> convert grayscale for : {root_dir}')
    
    # process dataset is main function in core.py that repeats for all other actions/tasks (flipV, flipH, brightness.... ect) except rotate
    process_dataset(root_dir = root_dir ,
                    output_dir = output_dir ,
                    func_img = convertGrayscale,  # func_img argument
                    func_label = copy_boxes, # func_label argument
                    debug = debug , 
                    verbose = verbose)
    
    if verbose: 
        print(f'>> adjContrast completed ')