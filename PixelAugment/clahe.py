import os
import cv2
import numpy as np


from .core import process_dataset, copy_boxes
# ---------------------------------------------------------
#  CLAHE / Histogram Equalization
# ---------------------------------------------------------
def clahe(img,factor=0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------
# Main entry from __init__ 
# ---------------------------------------------------------
def clahe_main (root_dir, output_dir, debug=False, verbose=True):
    if verbose: 
        print(f'>> convert grayscale for : {root_dir}')
    
    # process dataset is main function in core.py that repeats for all other actions/tasks (flipV, flipH, brightness.... ect) except rotate
    process_dataset(root_dir = root_dir ,
                    output_dir = output_dir ,
                    func_img = clahe,  # func_img argument
                    func_label = copy_boxes, # func_label argument
                    debug = debug , 
                    verbose = verbose,)
    
    if verbose: 
        print(f'>> adjContrast completed ')