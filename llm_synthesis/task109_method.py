import numpy as np

def method(grid):
    g = np.asarray(grid)
    H, W = g.shape
    
    # Find cross row (row full of same non-zero value)
    cross_row = cross_col = -1
    cross_color = 0
    for i in range(H):
        row = g[i]
        if np.all(row == row[0]) and row[0] != 0:
            cross_row = i
            cross_color = row[0]
            break
    for j in range(W):
        col = g[:, j]
        if np.all(col == col[0]) and col[0] != 0:
            cross_col = j
            break
    
    # Find which quadrant has the pattern
    quads = {
        'TL': g[:cross_row, :cross_col],
        'TR': g[:cross_row, cross_col+1:],
        'BL': g[cross_row+1:, :cross_col],
        'BR': g[cross_row+1:, cross_col+1:],
    }
    # Pick the non-empty one
    pattern = None
    for k, q in quads.items():
        if np.any(q != 0):
            pattern = q
            # orient as if it were TL
            if k == 'TR':
                pattern = pattern[:, ::-1]
            elif k == 'BL':
                pattern = pattern[::-1, :]
            elif k == 'BR':
                pattern = pattern[::-1, ::-1]
            break
    
    if pattern is None:
        pattern = quads['TL']
    
    # Convert pattern: non-zero -> cross_color
    p = np.where(pattern != 0, cross_color, 0)
    
    r, c = p.shape
    out = np.zeros((2*r, 2*c), dtype=g.dtype)
    out[:r, :c] = p
    out[:r, c:] = p[:, ::-1]
    out[r:, :c] = p[::-1, :]
    out[r:, c:] = p[::-1, ::-1]
    
    return out
