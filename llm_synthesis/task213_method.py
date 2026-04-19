def method(grid):
    import numpy as np
    H, W = grid.shape
    
    # Find the 5-rectangle
    rows5, cols5 = np.where(grid == 5)
    if len(rows5) == 0:
        return grid.copy()
    rh = rows5.max() - rows5.min() + 1
    rw = cols5.max() - cols5.min() + 1
    
    # Row lines: single non-0,non-5 color, touches left or right edge,
    # and color spans at least rw cells (so it's a real line, not a crossing)
    row_lines = []
    for r in range(H):
        row = grid[r]
        vals = set(int(x) for x in row) - {0, 5}
        if len(vals) == 1:
            color = vals.pop()
            count = int(np.sum(row == color))
            if (row[0] == color or row[-1] == color) and count >= rw:
                row_lines.append((r, color))
    
    # Column lines: analogous
    col_lines = []
    for c in range(W):
        col = grid[:, c]
        vals = set(int(x) for x in col) - {0, 5}
        if len(vals) == 1:
            color = vals.pop()
            count = int(np.sum(col == color))
            if (col[0] == color or col[-1] == color) and count >= rh:
                col_lines.append((c, color))
    
    if len(row_lines) >= len(col_lines):
        out = np.zeros((len(row_lines), rw), dtype=grid.dtype)
        for i, (_, color) in enumerate(row_lines):
            out[i, :] = color
        return out
    else:
        out = np.zeros((rh, len(col_lines)), dtype=grid.dtype)
        for i, (_, color) in enumerate(col_lines):
            out[:, i] = color
        return out
