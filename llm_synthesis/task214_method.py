import numpy as np

def method(grid):
    out = grid.copy()
    p1 = grid[:, 0:3]
    p2 = np.rot90(p1, k=-1)  # 90° clockwise
    p3 = np.rot90(p1, k=2)   # 180°
    out[:, 4:7] = p2
    out[:, 8:11] = p3
    return out
