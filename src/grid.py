
import numpy as np
from src.config import t_cold, t_hot, R_INNER, R_TANK, H


class PatternGenerator:

    def __init__(self, grid_width: int, grid_height: int):
        self.width = grid_width
        self.height = grid_height

    # -> np.ndarray is a type hint for the return value which is array
    def interference_pattern(self, cycles: float = 1.0) -> np.ndarray:
  
        # np.linspace(start, stop, n) = n evenly spaced numbers from start to
        # stop which includes both ends
        x = np.linspace(0, cycles * 2 * np.pi, self.width)
        y = np.linspace(0, cycles * 2 * np.pi, self.height)

        # indexing='ij' is method so output shape is strictly width, height
        X, Y = np.meshgrid(x, y, indexing="ij")

        pattern = 0.5 + 0.5 * np.cos(X) * np.sin(Y)
        # .astype(np.float64) converts to 64-bit floats 
        return pattern.astype(np.float64)

    def no_pattern(self, fill_value: float = 0.0) -> np.ndarray:
        # np.full(shape, value) makes an array of the given shape with every
        # element equal to value with dtype as the element type
        return np.full((self.width, self.height), fill_value, dtype=np.float64)


    #my old helper function
    def convection_init(self) -> np.ndarray:
        # np.zeros(shape) = array of that shape filled with 0.0.
        T = np.zeros((self.width, self.height), dtype=np.float64)

        # linear vertical temperature gradient
        for j in range(self.height):
            # t goes from t_cold at the top to t_hot at the bottom.
            T[:, j] = t_cold + (t_hot - t_cold) * (j / (self.height - 1))

        
        noise = np.random.uniform(-0.01, 0.01, size=(self.width - 2, self.height - 2))
        # += on a slice adds element-by-element directly into array
        T[1:-1, 1:-1] += noise

        return T


    #actual method being used 
    def create_tank_geometry(self):
       
        #know how this works by now
        x = np.arange(self.width, dtype=np.float64)
        y = np.arange(self.height, dtype=np.float64)
        I, J = np.meshgrid(x, y, indexing="ij")

        # find center again
        xc = (self.width - 1) / 2.0
        yc = (self.height - 1) / 2.0

        # find the center based grid coordinates  and sizeify them
        x_phys = (I - xc) * H
        y_phys = (J - yc) * H
        # distance of every cell from the center in cells computed for all cells
        r_grid = np.sqrt((I - xc)**2 + (J - yc)**2)

        #make the mask for inside and outside
        inside_tank = r_grid <= R_TANK
        inner_core  = r_grid <= R_INNER
        outer_rim   = r_grid >= R_TANK

        # main.py only keeps r_grid 
        return x_phys, y_phys, r_grid, inside_tank, inner_core, outer_rim



