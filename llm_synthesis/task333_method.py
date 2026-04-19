import numpy as np

def method(grid):
    out = grid.copy()
    H, W = grid.shape
    
    # Find 2x2 block of 3s
    r0 = c0 = None
    for r in range(H-1):
        for c in range(W-1):
            if (grid[r,c]==3 and grid[r+1,c]==3 and 
                grid[r,c+1]==3 and grid[r+1,c+1]==3):
                r0, c0 = r, c
                break
        if r0 is not None:
            break
    
    rows = [r0, r0+1]
    cols = [c0, c0+1]
    
    # Right: for each row in block, scan right
    for r in rows:
        for c in range(c0+2, W):
            if grid[r,c] != 0:
                color = grid[r,c]
                for cc in range(c0+2, c):
                    out[r,cc] = color
                break
    
    # Left
    for r in rows:
        for c in range(c0-1, -1, -1):
            if grid[r,c] != 0:
                color = grid[r,c]
                for cc in range(c+1, c0):
                    out[r,cc] = color
                break
    
    # Down
    for c in cols:
        for r in range(r0+2, H):
            if grid[r,c] != 0:
                color = grid[r,c]
                for rr in range(r0+2, r):
                    out[rr,c] = color
                break
    
    # Up
    for c in cols:
        for r in range(r0-1, -1, -1):
            if grid[r,c] != 0:
                color = grid[r,c]
                for rr in range(r+1, r0):
                    out[rr,c] = color
                break
    
    return out
