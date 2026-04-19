import numpy as np

def method(grid):
    out = grid.copy()
    h, w = grid.shape
    from collections import defaultdict
    pts = defaultdict(list)
    for r in range(h):
        for c in range(w):
            v = grid[r, c]
            if v != 0:
                pts[v].append((r, c))
    for color, lst in pts.items():
        if len(lst) == 2:
            (r1, c1), (r2, c2) = lst
            dr = r2 - r1
            dc = c2 - c1
            steps = max(abs(dr), abs(dc))
            sr = dr // steps if steps else 0
            sc = dc // steps if steps else 0
            for i in range(steps + 1):
                rr = r1 + sr * i
                cc = c1 + sc * i
                if out[rr, cc] == 0:
                    out[rr, cc] = color
    return out
