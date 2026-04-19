import numpy as np
from scipy.ndimage import label

def method(grid):
    grid = np.asarray(grid)
    best = None
    best_size = 0
    for color in np.unique(grid):
        if color == 0:
            continue
        mask = (grid == color)
        lbl, n = label(mask)
        for i in range(1, n + 1):
            comp = (lbl == i)
            size = comp.sum()
            if size > best_size:
                best_size = size
                rs, cs = np.where(comp)
                r0, r1 = rs.min(), rs.max()
                c0, c1 = cs.min(), cs.max()
                best = grid[r0:r1+1, c0:c1+1].copy()
                # zero out non-component cells within bbox
                local_mask = comp[r0:r1+1, c0:c1+1]
                out = np.where(local_mask, color, 0)
                best = out
    return best
