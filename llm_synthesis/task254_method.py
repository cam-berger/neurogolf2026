def method(grid):
    import numpy as np
    out = np.zeros_like(grid)
    h, w = grid.shape
    cols = []
    for c in range(w):
        cnt = int((grid[:, c] == 5).sum())
        if cnt > 0:
            cols.append((c, cnt))
    if not cols:
        return out
    longest = max(cols, key=lambda x: x[1])
    shortest = min(cols, key=lambda x: x[1])
    for c, cnt in cols:
        if c == longest[0]:
            color = 1
        elif c == shortest[0]:
            color = 2
        else:
            continue
        for r in range(h):
            if grid[r, c] == 5:
                out[r, c] = color
    return out
