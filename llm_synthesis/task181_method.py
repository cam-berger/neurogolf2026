def method(grid):
    import numpy as np
    g = grid.copy()
    
    r4, c4 = np.where(g == 4)
    r8, c8 = np.where(g == 8)
    
    r4_min, r4_max = r4.min(), r4.max()
    c4_min, c4_max = c4.min(), c4.max()
    r8_min, r8_max = r8.min(), r8.max()
    c8_min, c8_max = c8.min(), c8.max()
    
    # Determine hook direction from top row of the 4-shape
    top_row = g[r4_min, c4_min:c4_max+1]
    if top_row[0] == 4:
        direction = 'left'
    else:
        direction = 'right'
    
    # Use same column range as 4-shape for consistency
    c_min = min(c8_min, c4_min)
    c_max = max(c8_max, c4_max)
    
    shape8 = g[r8_min:r8_max+1, c_min:c_max+1]
    flipped = np.fliplr(shape8)
    
    h, w = flipped.shape
    if direction == 'left':
        target_c = c_min - w
    else:
        target_c = c_max + 1
    
    for i in range(h):
        for j in range(w):
            if flipped[i, j] != 0:
                rr = r8_min + i
                cc = target_c + j
                if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                    g[rr, cc] = flipped[i, j]
    
    return g
