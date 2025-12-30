#!/usr/bin/env python

import os

#from flipV import flipV_dataset
#from flipH import flipH_dataset
#from brightness import brightness_dataset
#from flipH import flipH 

from BBox import flipH, flipV, adjContrast, adjBrightness

datasetDir='train'          # input dir where all your training images and labels are (YOLO format) 
output_dir='trainAugmented' # results folder can also be the same name as dataset  


flipV(datasetDir , output_dir, debug=True ,verbose=True)
flipH(datasetDir , output_dir, debug=True ,verbose=True)
adjContrast(datasetDir , output_dir, debug=True ,verbose=True, factor=1.5)
adjContrast(datasetDir , output_dir, debug=True ,verbose=True, factor=0.2)
adjBrightness(datasetDir , output_dir, debug=True ,verbose=True, factor=90) #delta 20, positive increase , negative decrease 
adjBrightness(datasetDir , output_dir, debug=True ,verbose=True, factor=20) #delta 20, positive increase , negative decrease 

