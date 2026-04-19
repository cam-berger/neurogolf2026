def method(grid):
    import numpy as np
    H, W = grid.shape
    # find the "line" color (non-background)
    # background is most common color
    vals, counts = np.unique(grid, return_counts=True)
    bg = vals[np.argmax(counts)]
    line_color = vals[np.argmin(counts)] if len(vals) > 1 else bg
    # count full rows and columns of line_color
    full_rows = sum(1 for r in range(H) if np.all(grid[r, :] == line_color))
    full_cols = sum(1 for c in range(W) if np.all(grid[:, c] == line_color))
    return np.full((full_rows + 1, full_cols + 1), bg, dtype=grid.dtype)
