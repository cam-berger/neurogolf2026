def method(grid):
    import numpy as np
    A = grid[:, 0:4]
    B = grid[:, 5:9]
    C = grid[:, 10:14]
    out = np.zeros_like(A)
    out = np.where(C == 1, 1, out)
    out = np.where(B == 9, 9, out)
    out = np.where(A == 4, 4, out)
    return out
