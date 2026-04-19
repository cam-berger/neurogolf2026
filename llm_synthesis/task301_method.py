def method(grid):
    import numpy as np
    H, W = grid.shape
    colors = {}
    for v in np.unique(grid):
        if v == 0:
            continue
        colors[int(v)] = int((grid == v).sum())
    base = 8
    others = [(c, n) for c, n in colors.items() if c != base]
    others.sort(key=lambda x: -x[1])
    
    out = np.zeros_like(grid)
    out[H-1, :] = base
    row = H - 2
    for c, n in others:
        if row < 0:
            break
        out[row, W-n:] = c
        row -= 1
    return out
