#!/usr/bin/env python

from PixelAugment import adjContrast, adjBrightness, randHSV, convertGrayscale
from PixelAugment import addNoise, desaturate ,colorJitter, clahe

from GeoAugment import cropBoxes,flipH, flipV, rotateImg


datasetDir='./train'          # input dir where all your training images and labels are (YOLO format) 
output_dir='./trainAugmented' # results folder can also be the same name as dataset  


# flipV(datasetDir , output_dir, debug=True, verbose=True)
# flipH(datasetDir , output_dir, debug=True ,verbose=True)
# adjContrast(datasetDir , output_dir, debug=True ,verbose=True, factor=1.5)
# adjContrast(datasetDir , output_dir, debug=True ,verbose=True, factor=0.2)
# adjBrightness(datasetDir , output_dir, debug=True ,verbose=True, factor=90) #delta 20, positive increase , negative decrease 
# adjBrightness(datasetDir , output_dir, debug=True ,verbose=True, factor=20) #delta 20, positive increase , negative decrease 

convertGrayscale(datasetDir , output_dir, debug=True ,verbose=True)

# desaturate(datasetDir , output_dir, debug=True ,verbose=True, factor=0.5) # to 1
# rotateImg(datasetDir , output_dir, debug=True ,verbose=True, factor=30)
# rotateImg(datasetDir , output_dir, debug=True ,verbose=True, factor=146)
# addNoise(datasetDir , output_dir, debug=True ,verbose=True, factor=70) # anyting more then 2% noise is too much 
# cropBoxes(datasetDir , output_dir, verbose=True, factor=0.10) # crop for isolating objects


# colorJitter(datasetDir , output_dir, debug=True ,verbose=True, factor=[0.5, 0.6, 0.6]) # brightness , contrast , saturation

randHSV(datasetDir , output_dir, debug=True ,verbose=True, factor=[10, 0.3, 0.3]) # [h_range, s_range, v_range] Hue, Saturation, Brightness
clahe(datasetDir , output_dir, debug=True ,verbose=True) # CLAHE / Histogram Equalization

