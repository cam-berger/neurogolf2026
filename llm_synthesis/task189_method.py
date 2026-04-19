def method(grid):
    import numpy as np
    grid = np.asarray(grid)
    h, w = grid.shape
    
    # Find the row and column that are entirely 8s
    row8 = next(i for i in range(h) if np.all(grid[i] == 8))
    col8 = next(j for j in range(w) if np.all(grid[:, j] == 8))
    
    # Four regions divided by the 8-lines
    regions = {
        'tl': grid[0:row8, 0:col8],
        'tr': grid[0:row8, col8+1:w],
        'bl': grid[row8+1:h, 0:col8],
        'br': grid[row8+1:h, col8+1:w],
    }
    
    # Identify 2x2 key and 6x6 pattern
    key = None
    pattern = None
    for r in regions.values():
        if r.shape == (2, 2):
            key = r
        elif r.shape == (6, 6):
            pattern = r
    
    # Build output: each 3x3 quadrant of the pattern uses the color
    # from the corresponding cell of the key
    out = np.zeros((6, 6), dtype=int)
    for i in range(6):
        for j in range(6):
            if pattern[i, j] == 3:
                out[i, j] = key[i // 3, j // 3]
    return out
