#from .core import process_dataset
from .flipH import flipH_main as flipH
from .flipV import flipV_main as flipV
from .contrast import adjContrast_main as adjContrast
from .brightness import adjBrightness_main as adjBrightness
from .grayscale import convertGrayscale_main as convertGrayscale
from .desaturate import desaturate_main as desaturate
from .colorJitter import colorJitter_main as colorJitter
from .rotate import rotateImg_main as rotateImg
from .noise import addNoise_main as addNoise