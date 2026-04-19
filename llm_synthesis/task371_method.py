def method(grid):
    import numpy as np
    out = grid.copy()
    ones = np.argwhere(grid == 1)
    cr = (ones[0][0] + ones[1][0]) // 2
    cc = (ones[0][1] + ones[1][1]) // 2
    for dr, dc in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
        out[cr+dr, cc+dc] = 3
    return out
