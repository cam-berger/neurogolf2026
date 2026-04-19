import numpy as np
from scipy.ndimage import label, generate_binary_structure

def method(grid):
    grid = np.array(grid)
    colors = [c for c in np.unique(grid) if c != 0]
    structure = generate_binary_structure(2, 2)  # 8-connectivity
    
    best_color = None
    best_count = -1
    best_shape = None
    
    for c in colors:
        mask = (grid == c)
        labeled, n = label(mask, structure=structure)
        if n > best_count:
            best_count = n
            best_color = c
            ys, xs = np.where(labeled == 1)
            r0, r1 = ys.min(), ys.max()
            c0, c1 = xs.min(), xs.max()
            sub = grid[r0:r1+1, c0:c1+1].copy()
            # Zero-out other colors just in case
            sub = np.where(sub == c, c, 0)
            shape = np.zeros((3, 3), dtype=grid.dtype)
            h, w = sub.shape
            shape[:h, :w] = sub
            best_shape = shape
    
    return best_shape
