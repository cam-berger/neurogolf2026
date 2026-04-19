import numpy as np

def method(grid):
    g = np.array(grid)
    H, W = g.shape
    
    # Find position of 2
    pos2 = tuple(np.argwhere(g == 2)[0])
    
    # Find the 3 that is diagonally adjacent to 2 (the corner 3)
    corner3 = None
    for dr in (-1, 1):
        for dc in (-1, 1):
            r, c = pos2[0] + dr, pos2[1] + dc
            if 0 <= r < H and 0 <= c < W and g[r, c] == 3:
                corner3 = (r, c)
    
    # Compute row widths: 4 if row has non-zero, else 1
    row_widths = [4 if np.any(g[r] != 0) else 1 for r in range(H)]
    col_widths = [4 if np.any(g[:, c] != 0) else 1 for c in range(W)]
    
    # Row/col starting positions
    row_starts = [0]
    for w in row_widths[:-1]:
        row_starts.append(row_starts[-1] + w)
    col_starts = [0]
    for w in col_widths[:-1]:
        col_starts.append(col_starts[-1] + w)
    
    total_r = sum(row_widths)
    total_c = sum(col_widths)
    out = np.zeros((total_r, total_c), dtype=g.dtype)
    
    # Place 3-blocks at positions of 2 and corner3
    for (r, c) in [pos2, corner3]:
        rs, re = row_starts[r], row_starts[r] + row_widths[r]
        cs, ce = col_starts[c], col_starts[c] + col_widths[c]
        out[rs:re, cs:ce] = 3
    
    return out
