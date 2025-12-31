import os
import cv2
import numpy as np

from .core import process_dataset, copy_boxes

# ---------------------------------------------------------
# ADJUST BRIGHTNESS
# ---------------------------------------------------------
def adjBrightness(img, delta=20):
    """
    Adjust brightness by adding a constant value.
    delta > 0 → brighter
    delta < 0 → darker
    """
    img = img.astype(np.int16)
    img = img + delta
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img



def adjBrightness_main (root_dir, output_dir=root_dir, debug=False, verbose=True, factor=20):
    if verbose: 
        print(f'>> adjusting brightness for : {root_dir}')
    
    # process dataset is main function in core.py that repeats for all other actions/tasks (flipV, flipH, brightness.... ect) except rotate
    process_dataset(root_dir = root_dir ,
                    output_dir = output_dir ,
                    func_img = adjBrightness,  # func_img argument
                    func_label = copy_boxes, # func_label argument
                    debug = debug , 
                    verbose = verbose,
                    factor = factor)
    
    if verbose: 
        print(f'>> adjBrightness completed ')


