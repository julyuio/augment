import os
import cv2
import numpy as np

from .core import process_dataset, copy_boxes


def nochange(img, factor=0):
    return img



def checkBB (root_dir, output_dir='_', debug=False, verbose=True):
    if verbose: 
        print(f'>> Checking BB for : {root_dir} output={output_dir}')
    
    # process dataset is main function in core.py that repeats for all other actions/tasks (flipV, flipH, brightness.... ect) except rotate
    process_dataset(root_dir = root_dir ,
                    output_dir = output_dir ,
                    func_img = nochange,  # func_img argument
                    func_label = copy_boxes, # func_label argument
                    debug = debug , 
                    verbose = verbose,
                    factor = 0)
    
    if verbose: 
        print(f'>> adjBrightness completed ')


