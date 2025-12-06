# filtros.py
import numpy as np

ker_gaus = (1/16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
