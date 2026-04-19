def method(grid):
    import numpy as np
    g = np.asarray(grid)
    if np.array_equal(g, g[::-1]) or np.array_equal(g, g[:, ::-1]):
        return np.array([[1]])
    return np.array([[7]])
