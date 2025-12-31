import os
import cv2
import numpy as np

from .core import process_dataset, copy_boxes

# ---------------------------------------------------------
# ADD RANDOM GAUSSIAN NOISE
# ---------------------------------------------------------
def addNoise(img, factor=25):
    """
    Adds Gaussian noise to an image.
    mean: noise mean
    std: noise standard deviation (higher = stronger noise)
    """
    mean=0
    std=factor
    
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy



# ---------------------------------------------------------
# Main entry from __init__ 
# ---------------------------------------------------------
def addNoise_main (root_dir, output_dir, debug=False, verbose=True, factor=25):
    if verbose: 
        print(f'>> Add Noise for : {root_dir}')
    
    # process dataset is main function in core.py that repeats for all other actions/tasks (flipV, flipH, brightness.... ect) except rotate
    process_dataset(root_dir = root_dir ,
                    output_dir = output_dir ,
                    func_img = addNoise,  # func_img argument
                    func_label = copy_boxes, # func_label argument
                    debug = debug , 
                    verbose = verbose,
                    factor = factor)
    
    if verbose: 
        print(f'>> addNoise completed ')
