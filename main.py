#!/usr/bin/env python

#__________ TLDR ____________

from PixelAugment import adjContrast, adjBrightness, randHSV, convertGrayscale
from PixelAugment import addNoise, desaturate ,colorJitter, clahe
from GeoAugment import cropBoxes,flipH, flipV, rotateImg

datasetDir='./train'           
output_dir='./trainAugmented'   

#Pixel Augement examples
adjContrast(datasetDir , output_dir, debug=True ,verbose=True, factor=0.2) #factor 0.2 
adjBrightness(datasetDir , output_dir, debug=True ,verbose=True, factor=-40) #delta 20, positive increase , negative decrease 
convertGrayscale(datasetDir , output_dir, debug=True ,verbose=True)
desaturate(datasetDir , output_dir, debug=True ,verbose=True, factor=0.5) # to 1
colorJitter(datasetDir , output_dir, debug=True ,verbose=True, factor=[0.5, 0.6, 0.6]) # brightness , contrast , saturation
randHSV(datasetDir , output_dir, debug=True ,verbose=True, factor=[10, 0.3, 0.3]) # [h_range, s_range, v_range] Hue, Saturation, Brightness
clahe(datasetDir , output_dir, debug=True ,verbose=True) # CLAHE / Histogram Equalization
addNoise(datasetDir , output_dir, debug=True ,verbose=True, factor=3) # anyting more then 2% noise is too much and model accuracy will decrease, but test with 70 to see how it works/looks

#Geometric Augement examples
rotateImg(datasetDir , output_dir, debug=True ,verbose=True, factor=30) # 30 deg CW
rotateImg(datasetDir , output_dir, debug=True ,verbose=True, factor=146) # 146 deg CW
rotateImg(datasetDir , output_dir, debug=True ,verbose=True, factor=360-45) # 45 deg CCW
cropBoxes(datasetDir , output_dir, verbose=True, factor=0.10) # crop for isolating objects- 10% of the size of teh BBox
flipV(datasetDir , output_dir, debug=True, verbose=True)
flipH(datasetDir , output_dir, debug=True ,verbose=True)


#__________ End of TLDR_____________

# For more detailed examplanation look at examples.py 

# ____ TODO _____

# TODO: Blur and Motion Blur to be added 
# TODO: Cutout to improve resillience
# TODO: Resize to faster training lower accuracy
# TODO: OBB - add support for Oriented bounding boxes
# TODO: Polygones - add support for polygones for segmentation

# TODO: Shear - Add variability to perspective to help your model be more resilient to camera and subject pitch and yaw. usually 10 to -10 deg / horizontal vertical, anything above 15% not good.


#______ Copy ______

#Example of copy all files from Train to TrainAugmented: 

#datasetDir='./train'           
#output_dir='./trainAugmented'

import subprocess
subprocess.run(["rsync", "-rav", datasetDir + '/', output_dir])


