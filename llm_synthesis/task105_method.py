import numpy as np

def method(grid):
    out = grid.copy()
    ys, xs = np.where(grid == 1)
    if len(ys) == 0:
        return out
    r0, r1 = ys.min(), ys.max()
    c0, c1 = xs.min(), xs.max()
    
    # Identify edge rows/cols: those with >= 3 ones
    edge_rows = [r for r in range(r0, r1 + 1) if np.sum(grid[r, c0:c1+1] == 1) >= 3]
    edge_cols = [c for c in range(c0, c1 + 1) if np.sum(grid[r0:r1+1, c] == 1) >= 3]
    
    # Fill horizontal edges
    for r in edge_rows:
        for c in range(c0, c1 + 1):
            if out[r, c] == 0:
                out[r, c] = 2
    # Fill vertical edges
    for c in edge_cols:
        for r in range(r0, r1 + 1):
            if out[r, c] == 0:
                out[r, c] = 2
    return out
