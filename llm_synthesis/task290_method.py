import numpy as np

def method(grid):
    mask = grid != 0
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    sub = grid[r0:r1+1, c0:c1+1].copy()
    vals, counts = np.unique(sub, return_counts=True)
    # outer color has more cells, inner has fewer
    order = np.argsort(-counts)
    outer = vals[order[0]]
    inner = vals[order[1]]
    result = np.where(sub == outer, inner, outer)
    return result
