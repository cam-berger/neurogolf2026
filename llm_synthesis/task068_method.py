import numpy as np

def method(grid):
    grid = np.asarray(grid)
    out = np.zeros_like(grid)
    # Find the non-zero color appearing exactly once
    unique_color = None
    unique_pos = None
    for c in range(1, 10):
        positions = np.argwhere(grid == c)
        if len(positions) == 1:
            unique_color = c
            unique_pos = tuple(positions[0])
            break
    if unique_color is None:
        return out
    r, c = unique_pos
    H, W = grid.shape
    # Place 3x3 box of 2s centered on (r,c), clipped to grid
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = 2
    out[r, c] = unique_color
    return out
