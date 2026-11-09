import numpy as np
import pygame
from src.config import grid_width, grid_height, window_width, window_height

class Renderer:
    def __init__(self):
        """
       init for renderer and Look Up TAble.
        """
        self.lut = self.createThermalLUT()

    def createThermalLUT(self):
        """
        Precomputes a 256x3 uint8(meaning stored as integer 2^8 as bitps) RGB color lookup table maping 
         values [0.0, 1.0]  to a thermal y color gradient ( Blue ->  Green ->  Red).
        
        Returns:
            np.ndarray: Array of shape (256, 3) containing RGB color triplets.
        """
        lut = np.zeros((256, 3), dtype=np.uint8)
        
        # Lower Half (Indices 0..127): Deep Blue (0, 0, 128) -> Bright Green (0, 255, 0)
        lut[:128, 0] = 0                                            # Red component
        lut[:128, 1] = np.linspace(0, 255, 128, dtype=np.uint8)    # Green component
        lut[:128, 2] = np.linspace(128, 0, 128, dtype=np.uint8)    # Blue component
        
        # Upper Half (Indices 128..255): Bright Green (0, 255, 0) -> Bright Red (255, 0, 0)
        lut[128:, 0] = np.linspace(0, 255, 128, dtype=np.uint8)    # Red component
        lut[128:, 1] = np.linspace(255, 0, 128, dtype=np.uint8)    # Green component
        lut[128:, 2] = 0                                            # Blue component
        
        return lut

    def render_field(self, data):
        """
        Converts a 2D float NumPy array of the pattern into a pygame surface
        
        Parameters:
            data (np.ndarray): 2D array of shape (GRID_WIDTH, GRID_HEIGHT) 
                              containing float values in the range [0.0, 1.0].
                              
        Returns:
            pygame.Surface: High-resolution Pygame surface ready to blit onto display window.
        """
        # keep values between 0.0 and 1.0 to prevent bad indexing
        clippedData = np.clip(data, 0.0, 1.0)
        
        # scale floats [0.0, 1.0] to integer indices [0, 255] for the color look up
        indices = (clippedData * 255).astype(np.uint8)
        
        #  transforms shape (W, H) into shape (W, H, 3) using lookup table
        rgbArray = self.lut[indices]
        
        # convert the 3D NumPy array directly into a Pygame Surface
        smallSurface = pygame.surfarray.make_surface(rgbArray)
        
        # 5. scale the low-res surface to fill display window cleanly
        scaledSurface = pygame.transform.smoothscale(
            smallSurface, (window_width, window_height)
        )
        
        return scaledSurface