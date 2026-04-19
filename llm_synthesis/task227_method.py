def method(grid):
    import numpy as np
    h = grid.shape[0] // 2
    top = grid[:h]
    bot = grid[h:]
    out = np.where((top == 0) & (bot == 0), 2, 0)
    return out
