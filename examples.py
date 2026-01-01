#!/usr/bin/env python


from PixelAugment import adjContrast, adjBrightness, randHSV, convertGrayscale
from PixelAugment import addNoise, desaturate ,colorJitter, clahe
from GeoAugment import cropBoxes,flipH, flipV, rotateImg


datasetDir='./train'          # input dir where all your training images and labels are (YOLO format) 
output_dir='./trainAugmented' # new dataset results folder can also be the same name as dataset  


# All functions if debug=True it will create a directory with '_debug' showing the 
# end-result with bounding boxes or polygons. This is not to be used in the training
# just for visualization only


# ____ Examples of PixelAugment _____

# PixelAugement is only looking at the pixels and no other transformations are applied 
# (pixels stay in the same place)

# Contrast  factor > 1.0 → higher contrast  and   factor < 1.0 → lower contrast
adjContrast(datasetDir , output_dir) # default factor is factor=1.5
# or
adjContrast(datasetDir , output_dir, debug=True ,verbose=True, factor=0.2)


# Brightness  factor > 0 → brighter and factor < 0 → darker, default factor=20 
adjBrightness(datasetDir , output_dir) #delta 20, positive increase , negative decrease
# or 
adjBrightness(datasetDir , output_dir, debug=True ,verbose=True, factor=-40) #delta 20, positive increase , negative decrease 


# Convert to Grayscale,  no factor
convertGrayscale(datasetDir , output_dir, debug=True ,verbose=True)


# Desaturate - reduce saturation by a factor. Default factor=0.5, max 1
desaturate(datasetDir , output_dir, debug=True ,verbose=True, factor=0.5) # to 1


# Color Jitter - random values between max and min for brightness , contrast , saturation
colorJitter(datasetDir , output_dir, debug=True ,verbose=True, factor=[0.5, 0.6, 0.6]) # brightness , contrast , saturation


#randomize HSV - improves robusteness, factor is a list of values factor = [h_range, s_range, v_range]
#for example for hue is a random value between -hrange to h_range
#default values are [10, 0.3, 0.3]  Hue, Saturation, Brightness
randHSV(datasetDir , output_dir, debug=True ,verbose=True, factor=[10, 0.3, 0.3]) # [h_range, s_range, v_range] Hue, Saturation, Brightness


#CLAHE or Histogram Equalization
clahe(datasetDir , output_dir, debug=True ,verbose=True) # CLAHE / Histogram Equalization


#Add random noise - can be used to simulate low quality sensors in low light or others..
addNoise(datasetDir , output_dir, debug=True ,verbose=True, factor=3) # anyting more then 2% noise is too much and model accuracy will decrease, but test with 70 to see how it works/looks


# ____ Examples of GeoAugment _____

#Geometric transformations like rotate , flip, crop and others usually also require transformation of the bounding boxes of polygons

#Rotate - rotate image by an agle theta specified as factor. It rotates the image withough cropping 
#Rotation does not preserves the Hight and width of the image as the corners need room to rotate/expand.
#Factor = theta the angle of rotation clockwise (CW). If CCW rotation is needed then just use 360deg-theta. 
rotateImg(datasetDir , output_dir, debug=True ,verbose=True, factor=30)#  30 deg CW
rotateImg(datasetDir , output_dir, debug=True ,verbose=True, factor=360-45)# 45 deg CCW

#Crop - Crops and saves all bounding boxes factor is % of teh size of the bounding box
#1 crop per bounding box  
cropBoxes(datasetDir , output_dir, verbose=True, factor=0.10) # crop for isolating objects

#Flip vertical - flips along the vertical axes
flipV(datasetDir , output_dir, debug=True, verbose=True)

#Flip Horizontal - flips along the horizontal axes 
flipH(datasetDir , output_dir, debug=True ,verbose=True)


#______ Copy ______

#Example of copy all files from Train to TrainAugmented: 

#datasetDir='./train'           
#output_dir='./trainAugmented'

import subprocess
subprocess.run(["rsync", "-rav", datasetDir + '/', output_dir])



