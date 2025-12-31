import os
import cv2
import numpy as np


from .core import process_dataset, copy_boxes
# ---------------------------------------------------------
# Color Jitter (Brightness / Contrast / Saturation / Hue)
# ---------------------------------------------------------
def randHSV(img,factor=[10, 0.3, 0.3]):
    h_range = factor[0]
    s_range = factor[1]
    v_range = factor[2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Random hue shift
    hsv[...,0] += np.random.uniform(-h_range, h_range)

    # Random saturation scaling
    hsv[...,1] *= (1 + np.random.uniform(-s_range, s_range))

    # Random brightness scaling
    hsv[...,2] *= (1 + np.random.uniform(-v_range, v_range))

    # Clip and convert back
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)





# ---------------------------------------------------------
# Main entry from __init__ 
# ---------------------------------------------------------
def randHSV_main (root_dir, output_dir, debug=False, verbose=True, factor=[10, 0.3, 0.3]):
    if verbose: 
        print(f'>> convert grayscale for : {root_dir}')
    

    # process dataset is main function in core.py that repeats for all other actions/tasks (flipV, flipH, brightness.... ect) except rotate
    process_dataset(root_dir = root_dir ,
                    output_dir = output_dir ,
                    func_img = randHSV,  # func_img argument
                    func_label = copy_boxes, # func_label argument
                    debug = debug , 
                    verbose = verbose,
                    factor = factor)
    
    if verbose: 
        print(f'>> adjContrast completed ')