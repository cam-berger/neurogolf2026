import numpy as np

def method(grid):
    g = np.array(grid)
    out = g.copy()
    
    # Find bounding box of 2s
    twos = np.argwhere(g == 2)
    if len(twos) == 0:
        # just clear 5s
        out[out == 5] = 0
        return out
    top_row = twos[:, 0].min()
    bot_row = twos[:, 0].max()
    left_col = twos[:, 1].min()
    right_col = twos[:, 1].max()
    
    # Clear all 5s first
    fives = np.argwhere(g == 5)
    out[out == 5] = 0
    
    # For each 5 cell outside the box, reflect across nearest box edge
    for r, c in fives:
        nr, nc = r, c
        if r < top_row:
            nr = 2 * top_row - r
        elif r > bot_row:
            nr = 2 * bot_row - r
        if c < left_col:
            nc = 2 * left_col - c
        elif c > right_col:
            nc = 2 * right_col - c
        if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
            if out[nr, nc] == 0:
                out[nr, nc] = 5
    
    return out
