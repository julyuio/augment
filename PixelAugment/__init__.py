#from .core import process_dataset

from .contrast import adjContrast_main as adjContrast
from .brightness import adjBrightness_main as adjBrightness
from .grayscale import convertGrayscale_main as convertGrayscale
from .desaturate import desaturate_main as desaturate
from .colorJitter import colorJitter_main as colorJitter
from .noise import addNoise_main as addNoise
from .randomizeHSV import randHSV_main as randHSV
from .clahe import clahe_main as clahe
