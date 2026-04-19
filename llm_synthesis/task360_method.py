import numpy as np

def method(grid):
    grid = np.asarray(grid)
    # Find separator column (filled with 5s)
    sep = None
    for c in range(grid.shape[1]):
        if np.all(grid[:, c] == 5):
            sep = c
            break
    left = grid[:, :sep]
    right = grid[:, sep+1:]
    # Make same width
    w = max(left.shape[1], right.shape[1])
    if left.shape[1] < w:
        left = np.pad(left, ((0,0),(w-left.shape[1],0)))
    if right.shape[1] < w:
        right = np.pad(right, ((0,0),(0,w-right.shape[1])))
    right_rev = right[:, ::-1]
    out = np.where(left != 0, left, right_rev)
    return out
