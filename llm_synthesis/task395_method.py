def method(grid):
    import numpy as np
    top = grid[:3]
    bot = grid[3:]
    out = np.where((top == 0) & (bot == 0), 2, 0)
    return out
