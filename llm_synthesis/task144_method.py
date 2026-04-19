def method(grid):
    import numpy as np
    top = grid[0:4]
    bot = grid[5:9]
    out = np.where((top == 0) & (bot == 0), 3, 0)
    return out
