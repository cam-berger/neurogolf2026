import numpy as np

def method(grid):
    H, W = grid.shape
    ys, xs = np.where(grid != 0)
    colors = [(y, grid[y, x]) for y, x in zip(ys, xs)]
    colors.sort()
    top = colors[0][1]
    bot = colors[1][1]
    out = np.zeros((H, W), dtype=grid.dtype)
    # top color
    out[0, :] = top
    out[2, :] = top
    out[1, 0] = top; out[1, -1] = top
    out[3, 0] = top; out[3, -1] = top
    out[4, 0] = top; out[4, -1] = top
    # bottom color
    out[7, :] = bot
    out[9, :] = bot
    out[5, 0] = bot; out[5, -1] = bot
    out[6, 0] = bot; out[6, -1] = bot
    out[8, 0] = bot; out[8, -1] = bot
    return out
