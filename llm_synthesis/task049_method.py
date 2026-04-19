import numpy as np

def method(grid):
    h, w = grid.shape
    best = None
    best_area = None
    for color in np.unique(grid):
        if color == 0:
            continue
        ys, xs = np.where(grid == color)
        if len(ys) == 0:
            continue
        r0, r1 = ys.min(), ys.max()
        c0, c1 = xs.min(), xs.max()
        sub = grid[r0:r1+1, c0:c1+1]
        # Check if bounding box is a solid rectangle of this color
        if np.all(sub == color):
            area = sub.size
            if best_area is None or area < best_area:
                best_area = area
                best = sub.copy()
    return best
