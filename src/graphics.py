#
#     128x 128 array of floats turns to  (128, 128, 3) array of RGB values
#     as a pygame image, stretched to 800 x 800


import numpy as np
import pygame
from src.config import window_width, window_height

class Renderer:
    def __init__(self):
        self.lut = self.createThermalLUT()

    def createThermalLUT(self):
        

        # uint8 = unsigned 8-bit integer a whole number from 0 to 255 or 2^8
        # look up table with 256 rows (one per colour), 3 columns (R, G, B).
        lut = np.zeros((256, 3), dtype=np.uint8)
        
        # lower half is blue (0, 0, 128) to  green (0, 255, 0)
        # lut[:128, 0] means rows 0 through 127, column 0 
        # np.linspace(0, 255, 128) = 128 evenly spaced numbers from 0 to 255

        lut[:128, 0] = 0                                            # Red 
        lut[:128, 1] = np.linspace(0, 255, 128, dtype=np.uint8)    # Green 
        lut[:128, 2] = np.linspace(128, 0, 128, dtype=np.uint8)    # Blue 
        
        # upper half is green (0, 255, 0) to red(255, 0, 0)
        lut[128:, 0] = np.linspace(0, 255, 128, dtype=np.uint8)    # Red component
        lut[128:, 1] = np.linspace(255, 0, 128, dtype=np.uint8)    # Green component
        lut[128:, 2] = 0                                            # Blue component
        
        return lut

    # data  is the  array to draw.
    # inside_tank = is just inside tank or not 
    # vmin, vmax  = the data value that should map to blue (vmin) and to red (vmax)
    def render_field(self, data: np.ndarray, inside_tank: np.ndarray = None,
                     vmin: float = 0.0, vmax: float = 1.0) -> pygame.Surface:

        
        # rescale so vmin -> 0 and vmax -> 1 (every element at once).
        normData = (data - vmin) / (vmax - vmin)
        # force everything into [0, 1] in case some values fell outside.
        clippedData = np.clip(normData, 0.0, 1.0)
        # turn each value into a row number of the table, 0..255.
        # .astype(np.uint8) converts floats to whole numbers 
        indices = (clippedData * 255).astype(np.uint8)

        #feedd them to the lookup table and get a 3d array
        rgbArray = self.lut[indices].copy()

        #set inside to nothing
        if inside_tank is not None:
            rgbArray[~inside_tank] = [20, 24, 32]

        # turn the array into a pygame image which pygame reads the
        #  as [x, y, colour]
        smallSurface = pygame.surfarray.make_surface(rgbArray)
        # Stretch the 128 x 128 image up to 800 x 800 but not blocky
        return pygame.transform.scale(smallSurface, (window_width, window_height))
