import numpy as np

def method(grid):
    g = np.array(grid)
    H, W = g.shape
    out = np.zeros_like(g)
    
    # Find the 2x2 block of 8s
    ys, xs = np.where(g == 8)
    r0, c0 = ys.min(), xs.min()
    
    # Center of the 2x2 block (between rows r0,r0+1 and cols c0,c0+1)
    # For each non-zero, non-8 pixel, determine which quadrant relative to block
    for r in range(H):
        for c in range(W):
            v = g[r, c]
            if v == 0 or v == 8:
                continue
            # Determine quadrant
            dr = 0 if r <= r0 else 1  # top or bottom
            dc = 0 if c <= c0 else 1  # left or right
            # Actually: r < r0+0.5 means top; use r <= r0 for top
            if r < r0 + 1:
                dr = 0
            else:
                dr = 1
            if c < c0 + 1:
                dc = 0
            else:
                dc = 1
            out[r0 + dr, c0 + dc] = v
    
    return out
