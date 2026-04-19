import numpy as np

def method(grid):
    g = grid.copy()
    for i in range(g.shape[0]):
        if np.all(g[i, :] == 0):
            g[i, :] = 2
    for j in range(g.shape[1]):
        if np.all((g[:, j] == 0) | (g[:, j] == 2)):
            g[:, j] = 2
    return g
