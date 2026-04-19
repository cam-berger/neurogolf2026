import numpy as np

def method(grid):
    g = np.asarray(grid)
    H, W = g.shape
    out = np.zeros_like(g)
    bottom = H - 1
    
    cols_2 = [c for c in range(W) if g[bottom, c] == 2]
    fives = [(r, c) for r in range(H) for c in range(W) if g[r, c] == 5]
    
    # Keep bottom row 2s
    for c in cols_2:
        out[bottom, c] = 2
    
    # Place all 5s
    for (r, c) in fives:
        out[r, c] = 5
    
    # For each bottom 2, draw its column (possibly bending around a 5)
    for c in cols_2:
        bend = None
        for (r, fc) in fives:
            if fc == c and r < bottom:
                bend = r
                break
        if bend is None:
            for r in range(bottom):
                if out[r, c] == 0:
                    out[r, c] = 2
        else:
            # Column c fills from bend+1 down to bottom
            for r in range(bend + 1, bottom + 1):
                if out[r, c] == 0:
                    out[r, c] = 2
            # Column c+1 fills from top down to bend+1
            if c + 1 < W:
                for r in range(0, bend + 2):
                    if out[r, c + 1] == 0:
                        out[r, c + 1] = 2
    
    return out
