import numpy as np

def method(grid):
    grid = np.asarray(grid)
    H, W = grid.shape
    out = np.zeros_like(grid)
    
    row_counts = np.count_nonzero(grid, axis=1)
    col_counts = np.count_nonzero(grid, axis=0)
    
    r_h = int(np.argmax(row_counts))  # horizontal arm row
    c_v = int(np.argmax(col_counts))  # vertical arm col
    
    # Horizontal arm cells
    h_cols = np.nonzero(grid[r_h, :])[0]
    h_vals = grid[r_h, h_cols]
    
    # Vertical arm cells
    v_rows = np.nonzero(grid[:, c_v])[0]
    v_vals = grid[v_rows, c_v]
    
    h_start = int(h_cols[0])
    h_period = len(h_cols)
    h_pattern = list(h_vals)
    
    v_start = int(v_rows[0])
    v_period = len(v_rows)
    v_pattern = list(v_vals)
    
    # Fill horizontal arm row
    for c in range(W):
        out[r_h, c] = h_pattern[(c - h_start) % h_period]
    
    # Fill vertical arm col
    for r in range(H):
        out[r, c_v] = v_pattern[(r - v_start) % v_period]
    
    return out
