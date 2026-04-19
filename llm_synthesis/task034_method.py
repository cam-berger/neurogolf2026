import numpy as np

def method(grid):
    grid = np.asarray(grid)
    H, W = grid.shape
    out = np.zeros_like(grid)
    
    ys, xs = np.where(grid != 0)
    r1, r2 = ys.min(), ys.max()
    c1, c2 = xs.min(), xs.max()
    
    C = None
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            v = grid[r, c]
            if v != 0 and v != 2:
                C = int(v)
                break
        if C is not None:
            break
    
    # Keep original non-2 color cells
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            if grid[r, c] == C:
                out[r, c] = C
    
    # For each 2 cell, determine direction and draw 3-wide diagonal beam
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            if grid[r, c] == 2:
                dr = -1 if r == r1 else 1
                dc = -1 if c == c1 else 1
                k = 0
                while True:
                    nr = r + k * dr
                    nc_center = c + k * dc
                    if nr < 0 or nr >= H:
                        break
                    drew = False
                    for j in (-1, 0, 1):
                        nc = nc_center + j
                        if 0 <= nc < W:
                            out[nr, nc] = C
                            drew = True
                    if not drew and k > 0:
                        break
                    k += 1
    
    return out
