#!/usr/bin/env python

import os
from flipV import flipV_dataset
from flipH import flipH_dataset
from brightness import brightness_dataset

datasetDir='train'
AugmentedDir='trainAugmented'


flipH_dataset(root_dir = datasetDir , output_dir = AugmentedDir)
flipV_dataset(root_dir = datasetDir , output_dir = AugmentedDir)
