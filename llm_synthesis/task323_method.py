import numpy as np

def method(grid):
    out = grid.copy()
    H, W = grid.shape
    pos = np.argwhere(grid == 8)[0]
    r8, c8 = int(pos[0]), int(pos[1])
    
    patterns = [
        [(-1, 0), (-1, 0), (0, 1), (0, 1)],   # up-right staircase
        [(1, 0), (1, 0), (0, -1), (0, -1)],    # down-left staircase
    ]
    
    for pattern in patterns:
        r, c = r8, c8
        i = 0
        while True:
            dr, dc = pattern[i % 4]
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                out[nr, nc] = 5
                r, c = nr, nc
                i += 1
            else:
                break
    
    return out
