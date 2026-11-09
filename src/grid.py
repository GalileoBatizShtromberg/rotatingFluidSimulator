import numpy as np


class PatternGenerator:

    def __init__(self, grid_width: int, grid_height: int):
        self.width = grid_width
        self.height = grid_height

    def interference_pattern(self, cycles: float = 1.0) -> np.ndarray:
        """ interference pattern f(x, y) = 0.5 + 0.5 * sin(X) * cos(Y)
        """
        x = np.linspace(0, cycles * 2 * np.pi, self.width)
        y = np.linspace(0, cycles * 2 * np.pi, self.height)

        # indexing='ij' is method so output shape is strictly (grid_width, grid_height)
        X, Y = np.meshgrid(x, y, indexing="ij")

        pattern = 0.5 + 0.5 * np.sin(X) * np.cos(Y)
        return pattern.astype(np.float64)

