import numpy as np

def method(grid):
    g = np.asarray(grid)
    out = np.zeros_like(g)
    
    # Find the 2x2 block of 2s
    coords2 = np.argwhere(g == 2)
    r = coords2[:, 0].min()
    c = coords2[:, 1].min()
    # Copy block
    out[r:r+2, c:c+2] = 2
    
    # Process each 5
    fives = np.argwhere(g == 5)
    for r5, c5 in fives:
        if r5 < r:
            nr = r - 1
        elif r5 > r + 1:
            nr = r + 2
        else:
            nr = r5
        if c5 < c:
            nc = c - 1
        elif c5 > c + 1:
            nc = c + 2
        else:
            nc = c5
        out[nr, nc] = 5
    
    return out
