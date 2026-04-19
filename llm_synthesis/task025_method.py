import numpy as np

def method(grid):
    g = np.array(grid)
    h, w = g.shape
    
    # Find full rows (entirely one non-zero color)
    full_rows = {}
    for r in range(h):
        row = g[r]
        if row[0] != 0 and (row == row[0]).all():
            full_rows[r] = int(row[0])
    
    # Find full cols
    full_cols = {}
    for c in range(w):
        col = g[:, c]
        if col[0] != 0 and (col == col[0]).all():
            full_cols[c] = int(col[0])
    
    # Initialize output: only keep the full lines
    out = np.zeros_like(g)
    for r, color in full_rows.items():
        out[r, :] = color
    for c, color in full_cols.items():
        out[:, c] = color
    
    # Process stray cells
    for r in range(h):
        for c in range(w):
            v = int(g[r, c])
            if v == 0:
                continue
            if r in full_rows or c in full_cols:
                continue  # part of a line
            
            matched = False
            # Try to match with a horizontal line (full row)
            for lr, lcolor in full_rows.items():
                if lcolor == v:
                    if r < lr:
                        tr = lr - 1
                    else:
                        tr = lr + 1
                    if 0 <= tr < h:
                        out[tr, c] = v
                    matched = True
                    break
            if matched:
                continue
            # Try to match with vertical line (full col)
            for lc, lcolor in full_cols.items():
                if lcolor == v:
                    if c < lc:
                        tc = lc - 1
                    else:
                        tc = lc + 1
                    if 0 <= tc < w:
                        out[r, tc] = v
                    matched = True
                    break
            # If no match, the stray is removed (already zero in out)
    
    return out
