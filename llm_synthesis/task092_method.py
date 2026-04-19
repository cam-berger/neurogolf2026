import numpy as np

def method(grid):
    out = grid.copy()
    H, W = out.shape
    
    # Group positions by color (excluding 0)
    colors = {}
    for r in range(H):
        for c in range(W):
            v = grid[r, c]
            if v != 0:
                colors.setdefault(v, []).append((r, c))
    
    horizontals = []
    verticals = []
    
    for color, pts in colors.items():
        # find pairs with matching row or column
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                r1, c1 = pts[i]
                r2, c2 = pts[j]
                if r1 == r2:
                    horizontals.append((color, r1, min(c1, c2), max(c1, c2)))
                elif c1 == c2:
                    verticals.append((color, c1, min(r1, r2), max(r1, r2)))
    
    # Draw horizontals first
    for color, r, c_start, c_end in horizontals:
        for c in range(c_start, c_end + 1):
            out[r, c] = color
    
    # Then verticals (override intersections)
    for color, c, r_start, r_end in verticals:
        for r in range(r_start, r_end + 1):
            out[r, c] = color
    
    return out
