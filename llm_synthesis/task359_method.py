import numpy as np

def method(grid):
    h, w = grid.shape
    
    def mode_and_count(arr):
        vals, counts = np.unique(arr, return_counts=True)
        i = np.argmax(counts)
        return vals[i], counts[i]
    
    row_scores = [mode_and_count(grid[r])[1] / w for r in range(h)]
    col_scores = [mode_and_count(grid[:, c])[1] / h for c in range(w)]
    
    out = grid.copy()
    if np.mean(row_scores) >= np.mean(col_scores):
        # horizontal bands: each row is one color
        for r in range(h):
            m, _ = mode_and_count(grid[r])
            out[r, :] = m
    else:
        # vertical bands: each column is one color
        for c in range(w):
            m, _ = mode_and_count(grid[:, c])
            out[:, c] = m
    return out
