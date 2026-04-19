import numpy as np

def method(grid):
    g = np.array(grid)
    out = g.copy()
    
    twos = np.argwhere(g == 2)
    eights = np.argwhere(g == 8)
    
    r2_min, c2_min = twos.min(axis=0)
    r2_max, c2_max = twos.max(axis=0)
    r8_min, c8_min = eights.min(axis=0)
    r8_max, c8_max = eights.max(axis=0)
    
    # Determine shift direction
    dr, dc = 0, 0
    # Vertical gap?
    if r2_max < r8_min:  # 8 is below 2
        dr = r8_min - r2_max - 1
    elif r8_max < r2_min:  # 8 is above 2
        dr = r8_max - r2_min + 1
    
    # Horizontal gap?
    if c2_max < c8_min:  # 8 is right of 2
        dc = c8_min - c2_max - 1
    elif c8_max < c2_min:  # 8 is left of 2
        dc = c8_max - c2_min + 1
    
    # Clear original 2s
    out[g == 2] = 0
    # Place shifted 2s
    for r, c in twos:
        out[r + dr, c + dc] = 2
    
    return out
