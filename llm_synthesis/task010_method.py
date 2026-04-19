import numpy as np

def method(grid):
    out = grid.copy()
    h, w = grid.shape
    cols = []
    for c in range(w):
        cnt = int((grid[:, c] == 5).sum())
        if cnt > 0:
            cols.append((c, cnt))
    # sort by count descending; assign 1,2,3,4
    cols.sort(key=lambda x: -x[1])
    for rank, (c, _) in enumerate(cols, start=1):
        mask = out[:, c] == 5
        out[mask, c] = rank
    return out
