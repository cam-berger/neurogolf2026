def method(grid):
    import numpy as np
    out = np.zeros_like(grid)
    
    for center_val, marker_val in [(2, 3), (1, 7)]:
        pos = np.argwhere(grid == center_val)
        if len(pos) == 0:
            continue
        r, c = pos[0]
        out[r, c] = center_val
        # up
        if np.any(grid[:r, c] == marker_val):
            out[r-1, c] = marker_val
        # down
        if np.any(grid[r+1:, c] == marker_val):
            out[r+1, c] = marker_val
        # left
        if np.any(grid[r, :c] == marker_val):
            out[r, c-1] = marker_val
        # right
        if np.any(grid[r, c+1:] == marker_val):
            out[r, c+1] = marker_val
    
    return out
