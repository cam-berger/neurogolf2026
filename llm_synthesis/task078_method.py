import numpy as np

def method(grid):
    g = grid.copy()
    h, w = g.shape
    for c in range(w):
        col = g[:, c]
        k = int(np.sum(col == 2))
        if k == 0:
            continue
        # find first 0 after the initial run of 1s (top of hole)
        r0 = None
        seen_one = False
        for r in range(h):
            if col[r] == 1:
                seen_one = True
            elif col[r] == 0 and seen_one:
                r0 = r
                break
        if r0 is None:
            continue
        # clear all 2s in this column
        g[col == 2, c] = 0
        # place k 2s starting at r0 going down
        for i in range(k):
            rr = r0 + i
            if rr < h:
                g[rr, c] = 2
    return g
